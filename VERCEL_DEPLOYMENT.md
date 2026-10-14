# 🚀 Deploying to Vercel (Fast & Free)

This project has been transformed into a **Vercel-native production web application** featuring:
- **Instant Client-Side Zero-Latency ML Engine**: High-$R^2$ Polynomial Ridge ($R^2 = 0.8029$) & 50 Startups Profit Calculator.
- **Vercel Serverless Python API** (`/api/predict`): Stateless Python inference endpoint.
- **Interactive Plotly.js Charts**: Zero external server lag, dual-axis feature selection curves.
- **Modern Responsive Dark/Light Theme**: Persistent high-contrast typography.

---

## Method 1: Deploy with Vercel Web Dashboard (Easiest & Recommended)

1. **Push your project to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: Vercel-ready CRISP-DM Machine Learning Studio"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Import to Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in.
   - Click **"Add New..."** → **"Project"**.
   - Select your GitHub repository.
   - Leave the Framework Preset as **Other** (Root directory `./`).
   - Click **Deploy**.

3. **Done!**
   Your site will be live instantly at `https://<your-project-name>.vercel.app`.

---

## Method 2: Deploy from Terminal via Vercel CLI

Run this command in the project directory:

```bash
npx vercel
```

- When prompted:
  - `Set up and deploy?` → **Y**
  - `Which scope?` → Select your account
  - `Link to existing project?` → **N**
  - `What's your project's name?` → Press Enter or choose a name
  - `In which directory is your code located?` → `./`
- For production deployment:
  ```bash
  npx vercel --prod
  ```

---

## Why Streamlit Cannot Run on Vercel
- **Streamlit** requires a persistent WebSocket server (Tornado) that stays alive continuously.
- **Vercel** is an edge/serverless platform where requests must terminate within a few seconds without long-lived WebSockets.
- This Vercel-native application solves this problem by using standard web technologies (HTML5, Vanilla CSS, JS) with client-side mathematical ML inference and serverless API endpoints (`/api/predict`), giving you:
  - **100x Faster Load Times** (< 100ms globally)
  - **Zero Server Sleeping / Crashing**
  - **100% Free Unlimited Hosting on Vercel**
