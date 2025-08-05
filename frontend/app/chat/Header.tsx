'use client'
import React from "react";
import { useRouter } from "next/navigation";

const Header: React.FC = () => {
  const router = useRouter();

  const handleClearChat = () => {
    console.log("Clear Chat clicked"); // Replace with your clear logic
  };

  const handleLogout = () => {
  fetch("http://localhost:8000/auth/logout", {
    method: "GET",
    credentials: "include", // ✅ Required to send cookies
  })
    .then((res) => {
      if (!res.ok) throw new Error("Logout failed");
      return res.json();
    })
    .then(() => {
      window.location.href = "/";
    })
    .catch((err) => {
      console.error("Logout error:", err);
    });
};


  return (
    <header className="fixed top-0  z-20 w-full  py-6 bg-black md:bg-transparent ">
  <div className="w-fit flex  items-center px-4 sm:px-6 ">
    {/* Project Name */}
    <h1 className="text-xl font-light text-white ">
      EmbedMindAI
    </h1>

    {/* Right-side Buttons */}
    <div className="items-center gap-3 fixed right-2 hidden md:block">

      
      {/* Logout Button */}
      <button
        onClick={handleLogout}
        className=" font-light py-2 px-4 text-lg  text-white   rounded-full hover:bg-white/30  transition duration-200"
      >
        Logout
      </button>
    </div>
  </div>
</header>
  );
};

export default Header;
