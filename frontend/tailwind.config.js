/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          darkest: '#090d16',  // Deep midnight blue-black
          dark: '#0f172a',     // Slate-900
          card: '#1e293b',     // Slate-800
          border: '#334155',   // Slate-700
          text: '#f8fafc',     // Slate-50
          muted: '#94a3b8',    // Slate-400
          primary: '#6366f1',  // Indigo-500
          cyan: '#22d3ee',     // Cyan-400
          emerald: '#34d399',  // Emerald-400 (HBP / Benign)
          rose: '#f43f5e',     // Rose-500 (CaP / Cancer)
          violet: '#a78bfa'    // Violet-400 (Attention)
        }
      },
      backgroundImage: {
        'glass-radial': 'radial-gradient(circle, rgba(15, 23, 42, 0.8) 0%, rgba(9, 13, 22, 0.95) 100%)',
      }
    },
  },
  plugins: [],
}
