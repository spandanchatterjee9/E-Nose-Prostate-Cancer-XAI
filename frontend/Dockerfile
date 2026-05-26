FROM node:18-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package.json ./
RUN npm install

# Copy application files
COPY . .

# Build Next.js application
RUN npm run build

# Expose Next.js server port
EXPOSE 3000

# Start production server
CMD ["npm", "run", "start"]
