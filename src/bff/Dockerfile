FROM node:22-slim AS builder

WORKDIR /app

COPY package.json tsconfig.json ./
RUN npm install
COPY src/ src/
RUN npm run build

# ---- runtime ----
FROM node:22-slim

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev
COPY --from=builder /app/dist dist/

EXPOSE 3901
CMD ["node", "dist/server.js"]
