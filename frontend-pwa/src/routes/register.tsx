import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Lock, Mail, User } from "lucide-react";

export const Route = createFileRoute("/register")({
  component: RegisterPage,
});

function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  // ✅ Added 'async' here so you can use 'await'
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Client-side validation check
    if (!password || password.length < 6) {
      setError("Password is required and must be at least 6 characters.");
      return;
    }

    if (!name || !email) {
      setError("Please fill in all fields.");
      return;
    }

    try {
      const res = await fetch("/auth/register", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Registration failed");
        return;
      }

      // Successful registration
      window.location.href = "/";
    } catch (err) {
      setError("An unexpected error occurred. Please try again.");
    }
  }; // ✅ Added missing closing brace for the function

  return (
    <div className="min-h-screen w-full grid place-items-center px-4 relative overflow-hidden">
      <div className="relative w-full max-w-md">
        <div className="panel p-6 sm:p-8 bg-panel/40 backdrop-blur-xl border border-border rounded-3xl shadow-2xl">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold tracking-tight">Create Account</h1>
            <p className="mt-2 text-sm text-muted-foreground">Join EnviroSense AI platform</p>
          </div>

          <form onSubmit={handleRegister} className="space-y-5">
            {/* Name Input */}
            <div className="relative">
              <User className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-xl border border-border bg-panel/60 pl-11 pr-4 py-3.5 text-sm transition-all focus:outline-none focus:ring-2 focus:ring-clean/40"
                placeholder="Full Name"
              />
            </div>

            {/* Email Input */}
            <div className="relative">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl border border-border bg-panel/60 pl-11 pr-4 py-3.5 text-sm transition-all focus:outline-none focus:ring-2 focus:ring-clean/40"
                placeholder="Email"
              />
            </div>

            {/* Password Input */}
            <div className="relative">
              <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full rounded-xl border bg-panel/60 pl-11 pr-4 py-3.5 text-sm transition-all focus:outline-none focus:ring-2 ${
                  error && !password ? "border-poor focus:ring-poor/40" : "border-border focus:ring-clean/40"
                }`}
                placeholder="Password"
              />
            </div>

            {/* Error Message Display */}
            {error && (
              <div className="text-poor text-xs font-medium px-1 bg-poor/10 py-2 rounded-md border border-poor/20 text-center">
                {error}
              </div>
            )}

            {/* Register Button */}
            <button
              type="submit"
              disabled={!password || !email || !name}
              className="w-full inline-flex items-center justify-center rounded-xl bg-[#4ade80] hover:bg-[#22c55e] disabled:opacity-50 disabled:cursor-not-allowed text-[#052e16] px-4 py-3.5 text-sm font-bold shadow-lg transition-all active:scale-[0.98]"
            >
              Register
            </button>
          </form>

          <div className="mt-8 text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link to="/login" className="text-[#4ade80] font-semibold hover:underline">
              Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}