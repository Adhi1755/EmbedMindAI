import { useEffect, useState } from "react";

export default function ProgressDisplay() {
  const [messages, setMessages] = useState<string[]>([]);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws/progress");

    socket.onmessage = (event) => {
      setMessages((prev) => [...prev, event.data]);
    };

    return () => socket.close();
  }, []);

  return (
    <div className="bg-black text-white p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-2">🛠️ Processing Steps</h2>
      <ul className="space-y-1">
        {messages.map((msg, idx) => (
          <li key={idx} className="text-sm text-green-400">
            {msg}
          </li>
        ))}
      </ul>
    </div>
  );
}