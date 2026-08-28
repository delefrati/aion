import Fastify from "fastify";
import cors from "@fastify/cors";
import { z } from "zod";

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8900";
const port = parseInt(process.env.BFF_PORT || "3901", 10);

const app = Fastify({ logger: true });

await app.register(cors, { origin: true });

// ─── Schema ───

const ChatRequestSchema = z.object({
  message: z.string().min(1).max(4096),
  conversation_id: z.string().uuid().nullable().optional(),
});

type ChatRequest = z.infer<typeof ChatRequestSchema>;

// ─── Health ───

app.get("/health", async () => ({ status: "ok", service: "bff" }));

// ─── Chat (non-streaming) ───

app.post("/chat", async (req, reply) => {
  const parsed = ChatRequestSchema.safeParse(req.body);
  if (!parsed.success) {
    return reply.status(400).send({
      error: "validation_error",
      details: parsed.error.issues,
    });
  }

  const res = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsed.data),
  });

  const body = await res.json();
  return reply.status(res.status).send(body);
});

// ─── Chat Stream (SSE passthrough) ───

app.post("/chat/stream", async (req, reply) => {
  const parsed = ChatRequestSchema.safeParse(req.body);
  if (!parsed.success) {
    return reply.status(400).send({
      error: "validation_error",
      details: parsed.error.issues,
    });
  }

  const res = await fetch(`${BACKEND_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsed.data),
  });

  if (!res.ok || !res.body) {
    return reply.status(res.status).send({ error: "backend_error" });
  }

  // SSE passthrough — relay the backend stream directly
  reply.raw.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      reply.raw.write(decoder.decode(value, { stream: true }));
    }
  } finally {
    reply.raw.end();
  }
});

// ─── Start ───

app.listen({ port, host: "0.0.0.0" }).then(() => {
  console.log(`BFF listening on :${port}`);
});
