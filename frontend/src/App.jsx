import { useEffect, useRef, useState } from "react";

// Replace with your FastAPI backend WebSocket endpoint
const WEBSOCKET_URL = `wss://mryeti-signscribe.hf.space/ws`; // Use wss:// for secure WebSocket

function App() {
    const videoRef = useRef(null);
    const [output, setOutput] = useState({ char: "", prob: "", text: "" });
    const [ws, setWs] = useState(null);

    useEffect(() => {
        // Initialize WebSocket connection
        const websocket = new WebSocket(WEBSOCKET_URL);
        setWs(websocket);

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setOutput(data); // Update output state with received data
        };

        websocket.onopen = () => console.log("Connected to the backend!");
        websocket.onclose = () => console.log("Disconnected from the backend.");
        websocket.onerror = (error) => console.error("WebSocket error:", error);

        return () => {
            if (websocket) websocket.close();
        };
    }, []);

    useEffect(() => {
        const currentVideoRef = videoRef.current;

        async function setupCamera() {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
            });
            currentVideoRef.srcObject = stream;

            const sendFrame = () => {
                const canvas = document.createElement("canvas");
                canvas.width = currentVideoRef.videoWidth;
                canvas.height = currentVideoRef.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(currentVideoRef, 0, 0);

                canvas.toBlob((blob) => {
                    if (blob && ws && ws.readyState === WebSocket.OPEN) {
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const byteArray = new Uint8Array(reader.result);
                            ws.send(byteArray); // Send byte array over WebSocket
                        };
                        reader.readAsArrayBuffer(blob);
                    }
                }, "image/jpeg");
            };

            const intervalId = setInterval(sendFrame, 500); // Adjust interval as needed

            return () => clearInterval(intervalId);
        }

        setupCamera();

        return () => {
            if (currentVideoRef && currentVideoRef.srcObject) {
                currentVideoRef.srcObject
                    .getTracks()
                    .forEach((track) => track.stop());
            }
        };
    }, [ws]);

    return (
        <div className="bg-gray-900 min-h-screen flex flex-col items-center text-white w-full h-full">
            {/* Title Section */}
            <header className="text-4xl font-bold my-6">
                SignScribe: ASL Recognition
            </header>

            {/* Main Content */}
            <div className="flex w-[90%] justify-center px-2">
                {/* Camera Canvas */}
                <div className="flex-1 w-full h-full">
                    <video
                        ref={videoRef}
                        autoPlay
                        className="w-[78%] h-[78%] rounded-lg shadow-lg transform scale-x-[-1]"
                    />
                </div>

                {/* Image Display */}
                <div className="flex-1 max-w-lg h-full">
                    <img
                        src="Figure.jpg"
                        alt="Reference Figure"
                        className="w-full h-full object-cover rounded-lg shadow-lg"
                    />
                </div>
            </div>

            {/* Output Section */}
            <footer className="w-full flex justify-center fixed bottom-0 bg-gray-800 py-4 text-gray-400">
                <div className="w-[70%] flex flex-col">
                    <span className="text-base">Character: {output.char}</span>
                    <span className="text-base">Confidence: {output.prob}</span>
                    <span className="text-2xl text-white mt-2">
                        String: {output.text}
                    </span>
                </div>
            </footer>
        </div>
    );
}

export default App;
