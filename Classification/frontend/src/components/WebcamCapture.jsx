import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

const SOCKET_SERVER_URL = 'http://localhost:5001'; // Change this to your Flask server URL

function App() {
  const [socket, setSocket] = useState(null);
  const [stream, setStream] = useState(null);
  const videoRef = useRef();

  useEffect(() => {
    // Connect to the WebSocket server
    const newSocket = io(SOCKET_SERVER_URL);
    setSocket(newSocket);

    // Clean up function to disconnect from the WebSocket server
    return () => {
      newSocket.disconnect();
    };
  }, []);

  const captureFrame = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);

      const video = videoRef.current;
      video.srcObject = mediaStream;
      video.play();

      const captureInterval = setInterval(() => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg');
        sendFrame(frame);
      }, 1000); // Adjust the interval as needed

      return () => clearInterval(captureInterval);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const sendFrame = (frame) => {
    // Emit the video frame data to the server
    socket.emit('video_frame', { frame });
  };

  return (
    <div>
      <button onClick={captureFrame}>Start Video Feed</button>
      <video ref={videoRef} />
    </div>
  );
}

export default App;
