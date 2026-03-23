/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tag: {
          red: { bg: '#FEE2E2', text: '#DC2626', border: '#FECACA' },
          blue: { bg: '#DBEAFE', text: '#2563EB', border: '#BFDBFE' },
          green: { bg: '#D1FAE5', text: '#059669', border: '#A7F3D0' },
          yellow: { bg: '#FEF3C7', text: '#D97706', border: '#FDE68A' },
          purple: { bg: '#F3E8FF', text: '#7C3AED', border: '#E9D5FF' },
          orange: { bg: '#FFEDD5', text: '#EA580C', border: '#FED7AA' },
        }
      }
    },
  },
  plugins: [],
}