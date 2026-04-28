import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Lock, Mail, ArrowRight, Loader2 } from "lucide-react";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign In · EnviroSense AI" },
      { name: "description", content: "Sign in to the EnviroSense AI environmental intelligence console." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("operator@envirosense.ai");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch("/auth/login", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, remember }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Invalid credentials");
      }

      // ✅ Use navigate instead of window.location for a faster SPA transition
      const me = await fetch("/api/me", {
        credentials: "include",
      });
      
      if (me.ok) {
        navigate({ to: "/" });
      } else {
        setError("Session not created. Check backend cookie settings.");
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full grid place-items-center px-4 py-10 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute inset-0 grid-bg opacity-[0.06] pointer-events-none" />
      <div className="absolute -top-32 -right-32 h-80 w-80 rounded-full bg-clean/10 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-32 -left-32 h-80 w-80 rounded-full bg-cyan/10 blur-3xl pointer-events-none" />

      <div className="relative w-full max-w-md">
        <div className="flex flex-col items-center justify-center mb-6">
          <div className="grid place-items-center h-14 w-14 rounded-2xl bg-clean/10 border border-clean/30 shadow-[0_0_32px_oklch(0.78_0.18_150_/_0.35)]">
            <img src="/icon-512.png" alt="Logo" className="h-10 w-10 object-contain" />
          </div>
          <div className="mt-3 text-center leading-tight">
            <div className="font-semibold tracking-tight text-lg">EnviroSense AI</div>
            <div className="text-[11px] text-muted-foreground uppercase tracking-widest">Intelligence Console</div>
          </div>
        </div>

        <div className="panel panel-glow-clean p-6 sm:p-8 bg-panel/40 backdrop-blur-md border border-border rounded-3xl">
          <h1 className="text-xl font-semibold tracking-tight">Sign in to your console</h1>
          <p className="mt-1 text-xs text-muted-foreground">
            Edge-to-Cloud environmental intelligence. Authorised personnel only.
          </p>

          <form onSubmit={handleLogin} className="mt-6 space-y-4">
            {error && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-medium">
                {error}
              </div>
            )}

            <div>
              <label className="text-[11px] uppercase tracking-wider text-muted-foreground" htmlFor="email">Email</label>
              <div className="mt-1.5 relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  id="email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-lg border border-border bg-panel/60 pl-9 pr-3 py-2.5 text-sm font-mono focus:ring-2 focus:ring-clean/40 transition-all outline-none"
                  placeholder="you@example.com"
                />
              </div>
            </div>

            <div>
              <label className="text-[11px] uppercase tracking-wider text-muted-foreground" htmlFor="password">Password</label>
              <div className="mt-1.5 relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  id="password"
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg border border-border bg-panel/60 pl-9 pr-3 py-2.5 text-sm font-mono focus:ring-2 focus:ring-clean/40 transition-all outline-none"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <div className="flex items-center justify-between text-xs">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <div 
                  onClick={() => setRemember(!remember)}
                  className={`relative h-4 w-7 rounded-full border transition-colors ${remember ? "bg-clean/30 border-clean/50" : "bg-panel border-border"}`}
                >
                  <div className={`absolute top-0.5 h-3 w-3 rounded-full transition-transform ${remember ? "left-0.5 translate-x-3 bg-clean" : "left-0.5 bg-muted-foreground"}`} />
                </div>
                <span className="text-muted-foreground">Remember me</span>
                <input type="checkbox" className="sr-only" checked={remember} readOnly />
              </label>
              <Link to="/" className="text-clean hover:underline">Forgot password?</Link>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="group w-full inline-flex items-center justify-center gap-2 rounded-lg bg-clean text-primary-foreground px-4 py-2.5 text-sm font-semibold shadow-lg hover:bg-clean/90 transition-all disabled:opacity-70"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Enter Console"} 
              {!isLoading && <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />}
            </button>
          </form>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center"><span className="w-full border-t border-border" /></div>
            <div className="relative flex justify-center text-[10px] uppercase tracking-widest">
              <span className="bg-panel/40 px-2 text-muted-foreground">Or continue with</span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => window.location.href = "/auth/google"}
            className="w-full flex items-center justify-center gap-3 rounded-lg border border-border bg-panel/40 px-4 py-2.5 text-sm font-medium hover:bg-panel/60 transition-all"
          >
            <GoogleIcon /> Google
          </button>

          <div className="mt-4 text-sm text-center">
            Don’t have an account?{" "}
            <Link to="/register" className="text-clean hover:underline font-semibold">Register</Link>
          </div>

          <p className="mt-6 text-[10px] leading-relaxed text-muted-foreground border-t border-border pt-4">
            EnviroSense AI Intelligence Layer is for advisory purposes only.
          </p>
        </div>
      </div>
    </div>
  );
}

const GoogleIcon = () => (
  <svg className="h-4 w-4" viewBox="0 0 24 24">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg>
);