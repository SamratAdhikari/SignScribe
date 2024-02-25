import React, { useState } from "react";
import WebcamCapture from "./WebcamCapture"; // Assuming the WebcamCapture component is in a separate file

function LoginPage() {
  const [showWebcam, setShowWebcam] = useState(false);

  const handleLoginClick = () => {
    setShowWebcam(true);
  };

  return (
    <div className="bg-gray-100 flex justify-center items-center h-screen">
      {!showWebcam ? (
        <div className="bg-white p-8 rounded shadow-md w-96">
          <h2 className="text-2xl font-semibold mb-4">Login</h2>
          <form>
            <div className="mb-4">
              <label
                htmlFor="username"
                className="block text-gray-700 font-semibold mb-2"
              >
                Username
              </label>
              <input
                type="text"
                id="username"
                name="username"
                placeholder="Enter your username"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div className="mb-4">
              <label
                htmlFor="password"
                className="block text-gray-700 font-semibold mb-2"
              >
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                placeholder="Enter your password"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
              />
            </div>
            <button
              type="button"
              onClick={handleLoginClick}
              className="w-full bg-indigo-500 text-white py-2 px-4 rounded-md hover:bg-indigo-600 focus:outline-none focus:bg-indigo-600"
            >
              Login
            </button>
          </form>
        </div>
      ) : (
        <WebcamCapture />
      )}
    </div>
  );
}

export default LoginPage;
