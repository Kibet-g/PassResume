# Frontend Setup Instructions

## Prerequisites
- Node.js 18.0 or higher
- npm or yarn package manager

## Installation Steps

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```
   or
   ```bash
   yarn install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   or
   ```bash
   yarn dev
   ```

The frontend application will start on `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Features

- **Drag & Drop Upload:** Easy resume file upload with visual feedback
- **Real-time Analysis:** Instant ATS compliance scoring
- **Keyword Matching:** Compare resume against job descriptions
- **AI Suggestions:** Personalized improvement recommendations
- **Responsive Design:** Works on desktop and mobile devices

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls
- **React Dropzone** - File upload component

## Configuration

The frontend is configured to proxy API requests to the Flask backend running on `http://localhost:5000`. This is set up in `next.config.js`.

## Troubleshooting

1. **Port already in use:**
   - Next.js will automatically try the next available port
   - Or specify a different port: `npm run dev -- -p 3001`

2. **API connection issues:**
   - Make sure the Flask backend is running on port 5000
   - Check the proxy configuration in `next.config.js`

3. **Build errors:**
   - Clear Next.js cache: `rm -rf .next`
   - Reinstall dependencies: `rm -rf node_modules && npm install`