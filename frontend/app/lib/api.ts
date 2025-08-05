


export const uploadPDF = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to upload PDF");

  const data = await res.json();
  return data.message;
};


export const sendMessageToBackend = async (message: string): Promise<string> => {
  const formData = new FormData();
  formData.append("query", message);
  const response = await fetch("http://127.0.0.1:8000/ask", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  return data.answer;
};

