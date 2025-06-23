import React, { useState, useRef, useEffect } from "react";
import { Upload, Video, CheckCircle, XCircle, Download } from "lucide-react";

export default function VideoUploader() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoDuration, setVideoDuration] = useState(null);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("info");
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [imageUrls, setImageUrls] = useState([]);
  const [statsReady, setStatsReady] = useState(false);
  const [showImages, setShowImages] = useState(false);
  const xhrRef = useRef(null);
  const videoRef = useRef(null);

  const allowedTypes = ["video/mp4", "video/webm", "video/ogg", "video/x-msvideo"];
  const MAX_FILE_SIZE = 500 * 1024 * 1024;

  const isValidVideoPath = (path) =>
    typeof path === "string" && /^[\w\-./ áéíóúÁÉÍÓÚñÑ]+$/.test(path);

  const formatFileSize = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  const formatDuration = (seconds) =>
    `${Math.floor(seconds / 60)}:${(Math.floor(seconds % 60)).toString().padStart(2, "0")} min`;

  const handleFileChange = (e) => {
    setMessage("");
    setStatsReady(false);
    setImageUrls([]);
    setVideoDuration(null);
    setUploadProgress(0);
    const file = e.target.files[0];
    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
      setVideoFile(null);
      setMessageType("error");
      setMessage("Tipo de archivo no soportado. Sólo MP4, WebM, OGG o AVI.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setVideoFile(null);
      setMessageType("error");
      setMessage(`El archivo supera el tamaño máximo de ${MAX_FILE_SIZE / (1024 * 1024)}MB.`);
      return;
    }

    if (!isValidVideoPath(file.name)) {
      setVideoFile(null);
      setMessageType("error");
      setMessage("El nombre del archivo contiene caracteres no válidos.");
      return;
    }

    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    setMessageType("info");
    setMessage(`Archivo cargado: ${file.name} (${formatFileSize(file.size)})`);
  };

  const handleRemoveFile = () => {
    setVideoFile(null);
    setVideoPreview(null);
    setVideoDuration(null);
    setMessage("");
    setUploadProgress(0);
    setImageUrls([]);
    setStatsReady(false);
    document.getElementById("videoInput").value = "";
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    setIsProcessing(true);
    setMessageType("info");
    setMessage("Subiendo y procesando video...");

    const formData = new FormData();
    formData.append("video", videoFile);

    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr; 
    xhr.open("POST", "http://localhost:8000/process");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          setMessageType("success");
          setMessage("Procesamiento completado correctamente.");
          setStatsReady(true);

          fetch("http://localhost:8000/images")
            .then((res) => res.json())
            .then((data) => {
              if (Array.isArray(data.images)) {
                const urls = data.images.map((name) => `http://localhost:8000/images/${name}?v=${Date.now()}`);
                setImageUrls(urls);
              }
            })
            .catch(() => console.warn("No se pudieron cargar las imágenes."));
        } else {
          throw new Error(data.detail || "Error inesperado en el servidor.");
        }
      } catch (err) {
        setMessageType("error");
        setMessage(err.message || "Error durante la solicitud.");
      } finally {
        setIsProcessing(false);
        setUploadProgress(0);
      }
    };

    xhr.onerror = () => {
      setIsProcessing(false);
      setMessageType("error");
      setMessage("Fallo de red al subir el video.");
    };

    xhr.send(formData);
  };

  useEffect(() => {
    if (videoRef.current) {
      const handleLoadedMetadata = () => {
        setVideoDuration(formatDuration(videoRef.current.duration));
      };
      videoRef.current.addEventListener("loadedmetadata", handleLoadedMetadata);
      return () => {
        if (videoRef.current) {
          videoRef.current.removeEventListener("loadedmetadata", handleLoadedMetadata);
        }
      };
    }
  }, [videoPreview]);

  return (
    <div className="w-full flex justify-center px-4">
      <div className="max-w-5xl w-full bg-gray-900 rounded-2xl shadow-2xl p-8">
        
        <div className="text-gray-400 text-sm mb-6">
          <strong>Importante:</strong> Se recomienda que el video a procesar este grabado desde una cámara fija ubicada en la parte trasera de la pista, centrada respecto al ancho de esta y con una altura elevada. Este ángulo es necesario para maximizar la precisión del análisis estadístico.
        </div>

        <div className="text-gray-400 text-sm mb-6">
          <strong>Nota:</strong> El <span className="text-green-400">jugador 0</span> es el más cercano a la cámara, y el <span className="text-blue-400">jugador 1</span> el más alejado.
        </div>

        <div className="block cursor-pointer mb-6">
          <input
            id="videoInput"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            disabled={isProcessing}
            ref={(input) => {
              if (input) {
                input.style.opacity = 0;
                input.style.position = "absolute";
                input.style.width = 0;
                input.style.height = 0;
                input.style.pointerEvents = "none";
              }
            }}
          />
          <div
            onClick={() => document.getElementById("videoInput")?.click()}
            className="mt-4 flex flex-col items-center justify-center border-2 border-dashed border-gray-600 rounded-xl px-6 py-10 hover:bg-gray-800 transition"
          >
            <p className="text-xl font-semibold text-gray-200 mb-2 text-center">
              <span className="text-sm text-gray-400">
                Formatos permitidos de vídeo: MP4, WebM, OGG, AVI. <br />
                Tamaño máximo del vídeo permitido: 500MB. <br />
                El nombre del archivo debe contener solo letras, números, espacios, guiones o puntos.
              </span>
            </p>

            <p className="text-xl font-semibold text-gray-200 mb-2 text-center">
              <strong> Haz clic o arrastra un video aquí.<br /></strong>
            </p>
            <Upload className="h-10 w-10 text-green-400 hover:text-green-300 transition" />
          </div>
        </div>

        {videoFile && (
          <div className="mb-6 text-center">
            <div className="inline-flex items-center gap-2 bg-gray-800 rounded-lg px-4 py-3">
              <Video className="h-5 w-5 text-gray-300" />
              <span className="text-sm text-gray-300 truncate">
                {videoFile.name} ({formatFileSize(videoFile.size)})
                {videoDuration && ` - ${videoDuration}`}
              </span>
              <button
                onClick={handleRemoveFile}
                className="ml-auto text-red-400 hover:text-red-600 text-xl font-bold"
              >
                &times;
              </button>
            </div>
            <video
              src={videoPreview}
              controls
              ref={videoRef}
              className="mt-3 rounded-lg border border-gray-700 w-full max-w-xs mx-auto"
            />
          </div>
        )}

        {uploadProgress > 0 && (
          <div className="w-full bg-gray-700 rounded-full h-4 mb-4 overflow-hidden">
            <div
              className="bg-gradient-to-r from-green-400 to-green-600 h-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        )}

        {message && (
          <div
            className={`mb-4 flex items-center gap-2 px-4 py-3 rounded-lg border ${
              messageType === "error"
                ? "bg-red-100/10 border-red-500 text-red-400"
                : messageType === "success"
                ? "bg-green-100/10 border-green-500 text-green-400"
                : "bg-blue-100/10 border-blue-500 text-blue-400"
            }`}
          >
            {messageType === "success" && <CheckCircle className="h-5 w-5 text-green-400" />}
            {messageType === "error" && <XCircle className="h-5 w-5 text-red-400" />}
            <span>{message}</span>
          </div>
        )}

        {videoFile && (
          <button
            onClick={handleUpload}
            disabled={isProcessing}
            className="flex items-center justify-center gap-2 w-full bg-green-600 hover:bg-green-700 transition text-white font-semibold px-6 py-3 rounded-2xl shadow-lg disabled:opacity-50"
          >
            {isProcessing ? (
              <>
                <span className="animate-spin w-5 h-5 border-2 border-t-2 border-white rounded-full" />
                Procesando...
              </>
            ) : (
              <span>Extraer características</span>
            )}
          </button>
        )}

        {isProcessing && (
          <button
            onClick={() => {
              if (xhrRef.current) {
                xhrRef.current.abort();
                setIsProcessing(false);
                setUploadProgress(0);
                setMessageType("info");
                setMessage("Operación cancelada por el usuario.");
              }
            }}
            className="mt-4 flex items-center justify-center gap-2 w-full bg-red-600 hover:bg-red-700 transition text-white font-semibold px-6 py-3 rounded-2xl shadow-lg"
          >
            Cancelar operación
          </button>
        )}

        {statsReady && (
          <div className="mt-10 mb-6 text-center flex flex-col md:flex-row justify-center gap-10 flex-wrap">
            <a
              href="http://localhost:8000/download/statistics"
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-3 rounded-2xl shadow-lg transition"
              download
            >
              <Download className="w-5 h-5" />
              Descargar estadísticas (.zip)
            </a>
            <a
              href="http://localhost:8000/download/images"
              className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-2xl shadow-lg transition"
              download
            >
              <Download className="w-5 h-5" />
              Descargar imágenes (.zip)
            </a>
            <a
              href="http://localhost:8000/download/videos"
              className="inline-flex items-center gap-2 bg-yellow-500 hover:bg-yellow-600 text-white font-semibold px-6 py-3 rounded-2xl shadow-lg transition"
              download
            >
              <Download className="w-5 h-5" />
              Descargar videos (.zip)
            </a>
          </div>
        )}

        {imageUrls.length > 0 && (
          <div className="mt-8">
            <button
              onClick={() => setShowImages(!showImages)}
              className="w-full bg-gray-800 text-gray-100 font-semibold px-4 py-3 rounded-lg shadow hover:bg-gray-700 transition"
            >
              {showImages ? "Ocultar imágenes generadas ▲" : "Ver imágenes generadas ▼"}
            </button>

            {showImages && (
              <div className="mt-4 transition-all duration-300 ease-in-out">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {imageUrls.map((url, idx) => (
                    <div key={idx} className="bg-gray-800 rounded-lg p-4 flex justify-center items-center">
                      <img
                        src={url}
                        alt={`img-${idx}`}
                        className="rounded-lg border border-gray-700 w-full max-w-xs mx-auto object-contain"
                      />
                      <p className="mt-2 text-sm text-center text-gray-300 truncate">
                        {url.split("/").pop().split("?")[0]}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
      </div>
    </div>
  );
}
