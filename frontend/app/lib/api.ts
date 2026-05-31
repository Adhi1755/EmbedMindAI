
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const uploadPDF = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
    credentials: "include",
  });

  if (!res.ok) throw new Error("Failed to upload PDF");

  const data = await res.json();
  return data.message;
};

export const sendMessageToBackend = async (message: string): Promise<string> => {
  const formData = new FormData();
  formData.append("query", message);
  const response = await fetch(`${API_URL}/ask`, {
    method: "POST",
    body: formData,
    credentials: "include",
  });
  const data = await response.json();
  return data.answer;
};
