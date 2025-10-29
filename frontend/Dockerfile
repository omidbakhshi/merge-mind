# Development stage
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose the port
EXPOSE ${PORT:-3000}

# Start development server
CMD ["npm", "start"]