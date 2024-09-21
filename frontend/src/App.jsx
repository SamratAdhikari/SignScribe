import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import Figure from "./assets/Figure.jpg";
import "./App.css";

function App() {
    const videoRef = useRef(null);
    const [text, setText] = useState("");
    const [char, setChar] = useState("");
    const [prob, setProb] = useState("");
    const socket = io("https://signscribe-backend.onrender.com:8501");
    // Update with your backend address

    useEffect(() => {
        // Access webcam and set up video stream
        const getUserMedia = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                });
                videoRef.current.srcObject = stream;
            } catch (err) {
                console.error("Error accessing webcam: ", err);
            }
        };

        getUserMedia();
    }, []);

    const captureFrame = () => {
        const video = videoRef.current;
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.save();
        context.scale(-1, 1); // Flip horizontally
        context.translate(-canvas.width, 0); // Move canvas position
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.restore();

        const dataUrl = canvas.toDataURL("image/jpeg");
        sendFrameToBackend(dataUrl);
    };

    const sendFrameToBackend = (frame) => {
        socket.emit("frame", frame); // Emit the frame through the socket
    };

    useEffect(() => {
        const interval = setInterval(captureFrame, 300); // Capture every 300ms
        return () => clearInterval(interval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        socket.on("frame_processed", (data) => {
            console.log(data);
            if (data.status === "success") {
                setText(data.text);
                setChar(data.char);
                setProb(data.prob);
            }
        });

        return () => socket.off("frame_processed");
    }, [socket]);

    return (
        <div className="container">
            <h1 className="title">SignScribe: ASL Recognition</h1>
            <div className="canvas">
                <div>
                    <video
                        className="container-video"
                        ref={videoRef}
                        autoPlay
                    />
                    <div className="predictions">
                        <span>Character: {char}</span>
                        <span>Probability: {prob}</span>
                    </div>
                </div>

                <img src={Figure} className="container-image" />
            </div>

            <div className="output-text">
                <h2>Text: {text}</h2>
            </div>
        </div>
    );
}

export default App;
