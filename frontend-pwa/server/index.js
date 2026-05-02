import dotenv from "dotenv";
dotenv.config({ path: "/home/pragnya/EnviroSenseAI/.env" });

import express from "express";
import session from "express-session";
import passport from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import pkg from "pg";
import handler from "../dist/server/server.js";
import bcrypt from "bcryptjs";

const { Pool } = pkg;

const app = express();
const PORT = process.env.PORT || 4173;

// ================= BASIC =================
app.use(express.json());
app.set("trust proxy", 1);

// ================= DB =================
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
});

// Debug DB connection
pool.connect()
  .then(() => console.log("✅ DB Connected"))
  .catch(err => console.error("❌ DB ERROR:", err));

// ================= SESSION =================
app.use(session({
  secret: process.env.SESSION_SECRET || "fallback_secret",
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: false,      
    sameSite: "lax",   
    httpOnly: true,
  },
}));

app.use(passport.initialize());
app.use(passport.session());

// ================= PASSPORT =================
passport.serializeUser((user, done) => done(null, user.id));

passport.deserializeUser(async (id, done) => {
  try {
    const res = await pool.query(
      "SELECT * FROM users WHERE id=$1",
      [id]
    );
    done(null, res.rows[0]);
  } catch (err) {
    done(err, null);
  }
});

// ================= GOOGLE AUTH =================
passport.use(new GoogleStrategy({
  clientID: process.env.GOOGLE_CLIENT_ID,
  clientSecret: process.env.GOOGLE_CLIENT_SECRET,
  callbackURL: "https://envirosense-ai.duckdns.org/auth/google/callback",
}, async (accessToken, refreshToken, profile, done) => {
  try {
    const email = profile.emails[0].value;

    let result = await pool.query(
      "SELECT * FROM users WHERE email=$1",
      [email]
    );

    // ✅ EXISTING USER → MERGE
    if (result.rows.length > 0) {
      let user = result.rows[0];

      if (!user.google_id) {
        const updated = await pool.query(
          `UPDATE users
           SET google_id=$1, avatar=$2
           WHERE id=$3
           RETURNING *`,
          [profile.id, profile.photos[0].value, user.id]
        );

        user = updated.rows[0];
      }

      return done(null, user);
    }

    // 🆕 NEW USER
    const newUser = await pool.query(
      `INSERT INTO users (google_id, email, name, avatar)
       VALUES ($1,$2,$3,$4)
       RETURNING *`,
      [
        profile.id,
        email,
        profile.displayName,
        profile.photos[0].value,
      ]
    );

    return done(null, newUser.rows[0]);

  } catch (err) {
    return done(err, null);
  }
}));

// ================= ROUTES =================

// Google
app.get("/auth/google",
  passport.authenticate("google", { scope: ["profile", "email"] })
);

app.get("/auth/google/callback",
  passport.authenticate("google", { failureRedirect: "/login" }),
  (req, res) => res.redirect("/")
);

// ================= REGISTER =================
app.post("/auth/register", async (req, res) => {
  const { name, email, password } = req.body;

  try {
    if (!name || !email || !password) {
      return res.status(400).json({ error: "All fields required" });
    }

    if (!email.includes("@") || !email.includes(".")) {
      return res.status(400).json({ error: "Invalid email" });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: "Password too short" });
    }

    const existing = await pool.query(
      "SELECT * FROM users WHERE email=$1",
      [email]
    );

    const hashed = await bcrypt.hash(password, 10);

    // 🔥 EXISTING USER
    if (existing.rows.length > 0) {
      let user = existing.rows[0];

      // Google user → attach password
      if (!user.password) {
        const updated = await pool.query(
          `UPDATE users
           SET name=$1, password=$2
           WHERE id=$3
           RETURNING *`,
          [name, hashed, user.id]
        );

        user = updated.rows[0];

        return req.login(user, () => {
          res.json({ success: true });
        });
      }

      return res.status(400).json({
        error: "User already exists. Please login.",
      });
    }

    // 🆕 NEW USER
    const result = await pool.query(
      `INSERT INTO users (name, email, password)
       VALUES ($1,$2,$3)
       RETURNING *`,
      [name, email, hashed]
    );

    req.login(result.rows[0], () => {
      res.json({ success: true });
    });

  } catch (err) {
    console.error("REGISTER ERROR:", err);
    res.status(500).json({ error: "Register failed" });
  }
});

// ================= LOGIN =================
app.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    const result = await pool.query(
      "SELECT * FROM users WHERE email=$1",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ error: "User not found" });
    }

    const user = result.rows[0];

    if (!user.password) {
      return res.status(400).json({
        error: "Use Google login for this account",
      });
    }

    const valid = await bcrypt.compare(password, user.password);

    if (!valid) {
      return res.status(401).json({ error: "Invalid password" });
    }

    req.login(user, (err) => {
      if (err) {
        return res.status(500).json({ error: "Session failed" });
      }
      res.json({ success: true });
    });

  } catch (err) {
    console.error("LOGIN ERROR:", err);
    res.status(500).json({ error: "Login failed" });
  }
});

// ================= CURRENT USER =================
app.get("/api/me", (req, res) => {
  if (req.isAuthenticated()) {
    res.json({ user: req.user });
  } else {
    res.status(401).json({ user: null });
  }
});

// ================= LOGOUT =================
app.get("/logout", (req, res) => {
  req.logout(() => res.redirect("/login"));
});

// ================= SSR =================
app.use(async (req, res) => {
  const url = `http://${req.headers.host}${req.url}`;

  const request = new Request(url, {
    method: req.method,
    headers: req.headers,
    ...(req.method !== "GET" && req.method !== "HEAD"
      ? { body: req, duplex: "half" }
      : {}),
  });

  const response = await handler.fetch(request);

  res.writeHead(response.status, Object.fromEntries(response.headers));
  const body = await response.arrayBuffer();
  res.end(Buffer.from(body));
});

// ================= START =================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
