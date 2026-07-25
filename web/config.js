/* KneeGuard AI — frontend configuration.
 *
 * Leave this alone when the dashboard and the API are served together (running
 * locally, Hugging Face Spaces, Render, Docker). Same-origin is the default.
 *
 * Set it only when the frontend is hosted separately from the backend — a
 * static host like Vercel, Netlify or GitHub Pages cannot run the Python
 * server, so it needs to be told where the API lives:
 *
 *     window.KNEEGUARD_API_BASE = "https://your-backend.onrender.com";
 *
 * The backend must then allow this site's origin, via its own env var:
 *
 *     CORS_ALLOW_ORIGINS=https://your-site.vercel.app
 *
 * No trailing slash on either.
 */
window.KNEEGUARD_API_BASE = "";
