// Non sono necessarie importazioni esplicite di React, useState, ecc.
// quando si usa Babel standalone e React/ReactDOM via CDN,
// in quanto sono disponibili globalmente.

function WasteRecognizer() {
  // Destrutturazione degli hook di React per comodità (disponibili globalmente)
  const { useState, useRef, useEffect, useMemo } = React;

  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImageSrc, setProcessedImageSrc] = useState(null); // URL per l'immagine dal backend
  const [detectionResults, setDetectionResults] = useState(null); // Risultati delle rilevazioni dal backend
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(''); // Per i messaggi di stato all'utente
  const canvasRef = useRef(null); // Riferimento al canvas per disegnare

  // Definizione delle istruzioni di riciclo per diversi materiali
  // Assicurati che questi nomi corrispondano esattamente alle classi del tuo modello YOLO (in minuscolo)
  const recyclingTips = {
    'glass': 'Il vetro (bottiglie, vasetti) va nel contenitore del vetro. Rimuovi tappi e etichette se possibile. Non buttare ceramica o porcellana.',
    'plastic': 'La plastica (bottiglie, contenitori, imballaggi) va nel contenitore della plastica. Sciacqua e schiaccia per ridurre il volume.',
    'metal': 'Il metallo (lattine, scatolette, alluminio) va nel contenitore del metallo o multimateriale. Sciacqua leggermente se conteneva alimenti.',
    'paper': 'La carta (giornali, riviste, quaderni) va nel contenitore della carta. Assicurati che sia pulita e asciutta.',
    'cardboard': 'Il cartone (scatole, imballaggi) va nel contenitore della carta/cartone. Appiattisci le scatole e rimuovi nastri adesivi.',
    'biodegradable': 'Il materiale biodegradabile (avanzi di cibo, scarti vegetali) va nel contenitore dell\'umido organico/compost. Usa sacchetti compostabili.',
    'Rifiuto Generico': 'Per questo rifiuto non ci sono istruzioni specifiche di riciclo. Conferire nell\'indifferenziato o verificare le regole locali del tuo comune.'
  };

  // Gestisce la selezione del file immagine dall'utente
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedImage(file);
    setMessage(''); // Reset del messaggio di stato
    setDetectionResults(null); // Reset dei risultati di rilevamento
    setProcessedImageSrc(null); // Reset dell'immagine elaborata

    // Mostra un'anteprima immediata dell'immagine selezionata sul canvas
    // Questo è utile prima dell'invio al backend per dare feedback all'utente
    const reader = new FileReader();
    reader.onloadend = () => {
      // Disegna l'immagine di anteprima immediatamente sul canvas
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        const img = new Image();
        img.onload = () => {
          // Mantieni l'aspetto dell'immagine, ma ridimensionala per adattarsi al canvas/display
          const maxWidth = 600; // Massima larghezza desiderata per il display
          const maxHeight = 450; // Massima altezza desiderata per il display
          let width = img.width;
          let height = img.height;

          if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height);
            width *= ratio;
            height *= ratio;
          }

          canvas.width = width;
          canvas.height = height;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, width, height);
        };
        img.src = reader.result;
      }
    };
    reader.readAsDataURL(file);
  };

  // Gestisce il click sul pulsante "Rileva rifiuti"
  const handleDetectClick = async () => {
    if (!selectedImage) {
      setMessage("Seleziona un'immagine prima di rilevare.");
      return;
    }

    setLoading(true);
    setMessage("Rilevamento in corso...");
    setDetectionResults(null); // Cancella i risultati precedenti

    try {
      const formData = new FormData();
      // Il tuo backend Flask si aspetta la chiave "image" per il file caricato
      formData.append("image", selectedImage); 

      // ***** INSERISCI QUI L'URL DEL TUO BACKEND NGROK *****
      // Questo URL è TEMPORANEO e cambierà ogni volta che avvii il tunnel Ngrok in Colab!
      // Devi copiare l'URL stampato nel tuo notebook Colab.
      const URL = "https://huggingface.co/spaces/Belloctopus/Re-life/detect"; // Esempio da aggiornare!

      const response = await fetch(URL, {
        method: "POST",
        body: formData,
        // Non impostare 'Content-Type', il browser lo gestisce automaticamente con FormData
      });

      if (!response.ok) {
        // Se la risposta non è OK (es. status 400, 500), leggi il messaggio di errore dal backend
        const errorText = await response.text();
        throw new Error(`HTTP error! Status: ${response.status}. Message: ${errorText}`);
      }

      const data = await response.json();

      // ***** STRUTTURA ATTESA DELLA RISPOSTA DAL TUO BACKEND Flask: *****
      // {
      //   "processed_image_base64": "data:image/jpeg;base64,...", // Stringa base64 dell'immagine con i bounding box disegnati
      //   "detected_objects": [ // Array di oggetti rilevati
      //     {
      //       "class": "nome_classe",    // Es: "glass", "plastic"
      //       "confidence": 0.95,        // Score di confidenza (0.0 - 1.0)
      //       "bbox": [xmin, ymin, xmax, ymax] // Coordinate normalizzate (0.0 - 1.0)
      //     },
      //     { ... altro oggetto rilevato ... }
      //   ],
      //   "message": "Rilevamento completato!"
      // }
      // Assicurati che il tuo backend continui a restituire questa struttura!

      setProcessedImageSrc(data.processed_image_base64);
      setDetectionResults(data.detected_objects || []); // Assicurati che sia un array anche se vuoto
      setMessage(data.message || 'Rilevamento completato!');

    } catch (e) {
      setMessage("Errore nel rilevamento. Riprova. Controlla la console del browser (F12) per i dettagli.");
      console.error("Errore durante il rilevamento:", e);
      setProcessedImageSrc(null); // Resetta l'immagine se c'è un errore
      setDetectionResults(null); // Resetta i risultati se c'è un errore
    } finally {
      setLoading(false);
    }
  };

  // useEffect per disegnare l'immagine e le rilevazioni sul canvas
  // Si attiva quando `processedImageSrc` (l'immagine dal backend) o `detectionResults` cambiano
  useEffect(() => {
    if (!processedImageSrc) {
      // Pulisci il canvas se non c'è un'immagine processata da mostrare
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      // Imposta le dimensioni del canvas per corrispondere all'immagine restituita dal backend
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Pulisci il canvas prima di ridisegnare
      ctx.drawImage(img, 0, 0); // Disegna l'immagine elaborata

      // Disegna i bounding box e le etichette solo se i detectionResults sono validi
      if (detectionResults && Array.isArray(detectionResults) && detectionResults.length > 0) {
        detectionResults.forEach((obj) => {
          // Aggiungi controlli per assicurarti che le proprietà esistano
          if (!obj || !obj.bbox || obj.bbox.length !== 4 || obj.class === undefined || obj.confidence === undefined) {
              console.warn("Oggetto di rilevamento malformato, saltato:", obj);
              return;
          }

          // bbox formato [xmin, ymin, xmax, ymax], normalizzati tra 0 e 1
          const [xmin, ymin, xmax, ymax] = obj.bbox;
          const x = xmin * canvas.width;
          const y = ymin * canvas.height;
          const width = (xmax - xmin) * canvas.width;
          const height = (ymax - ymin) * canvas.height;

          const labelText = obj.class; // Usa 'class' come nome della proprietà
          const confidenceText = (obj.confidence * 100).toFixed(1);

          ctx.strokeStyle = "#e68387"; // Colore del bordo (salmone)
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          ctx.fillStyle = "#e68387"; // Sfondo testo colore salmone
          ctx.font = "bold 18px Arial"; // Testo più leggibile
          ctx.textBaseline = "top"; // Allineamento testo in alto
          
          const text = `${labelText} (${confidenceText}%)`;
          const textMetrics = ctx.measureText(text);
          const textWidth = textMetrics.width;
          const textHeight = 20; // Stima altezza testo per sfondo (può variare in base al font)

          // Disegna sfondo per il testo
          ctx.fillRect(x, y > textHeight ? y - textHeight - 5 : y, textWidth + 10, textHeight + 5);
          
          ctx.fillStyle = "white"; // Colore del testo
          ctx.fillText(text, x + 5, y > textHeight ? y - textHeight : y + 5);
        });
      }
    };
    img.src = processedImageSrc; // Carica l'immagine dal backend nel canvas
  }, [processedImageSrc, detectionResults]); // Dipendenze dell'useEffect

  // Genera la guida al riciclo in base agli elementi rilevati
  // useMemo per ricalcolare la guida solo quando i risultati di rilevamento cambiano
  const recyclingGuide = useMemo(() => {
    if (!detectionResults || !Array.isArray(detectionResults) || detectionResults.length === 0) return [];
    
    // Filtra gli oggetti validi prima di mappare le label
    // Converti la classe in minuscolo per il matching con recyclingTips
    const validDetectedObjects = detectionResults.filter(obj => obj && obj.class !== undefined);
    const detectedLabels = new Set(validDetectedObjects.map(obj => obj.class.toLowerCase())); // Modifica qui
    
    const guide = [];
    detectedLabels.forEach(label => {
      guide.push({
        label: label,
        tip: recyclingTips[label] || recyclingTips['Rifiuto Generico'] // Fallback a suggerimento generico
      });
    });
    return guide;
  }, [detectionResults]);

  // Il JSX che definisce l'interfaccia utente del componente
  return (
    <div className="flex flex-col items-center justify-center gap-6 p-4 w-full h-full max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold mb-4" style={{ color: '#e68387' }}>Rilevazione Rifiuti</h2>

      <div className="w-full max-w-md flex flex-col items-center gap-4 p-6 border-2 border-dashed border-gray-300 rounded-lg shadow-md bg-gray-50 hover:bg-gray-100 transition-colors duration-200">
        <label htmlFor="fileInput" className="cursor-pointer flex flex-col items-center justify-center p-4 w-full h-full">
          <span className="text-5xl text-gray-700 mb-2">🖼️</span> {/* Icona immagine */}
          <span className="text-xl font-semibold text-gray-700 text-center">Trascina o Clicca per Caricare Immagine</span>
          <span className="text-sm text-gray-500 mt-1">Accetta JPG, PNG</span>
        </label>
        <input id="fileInput" type="file" accept="image/jpeg, image/png" onChange={handleImageChange} className="hidden" />
      </div>

      {/* Mostra il canvas solo se c'è un'immagine selezionata o processata */}
      {(selectedImage || processedImageSrc) && (
          <canvas ref={canvasRef} className="max-w-full rounded-lg shadow-lg border" style={{ borderColor: '#e68387' }} />
      )}

      {/* Area messaggi */}
      {message && (
        <div className={`mt-2 p-2 rounded text-center text-sm font-medium ${message.includes('Errore') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
          {message}
        </div>
      )}

      <button
        onClick={handleDetectClick}
        disabled={loading || !selectedImage} // Disabilita se sta caricando o se nessuna immagine è stata selezionata
        className="mt-4 px-6 py-3 rounded bg-[#FA8072] text-white font-semibold hover:bg-[#e27366] disabled:bg-gray-400 transition"
      >
        {loading ? "Rilevamento in corso..." : "Rileva rifiuti"}
      </button>

      {/* Guida al Riciclo - Mostra solo se ci sono risultati di rilevamento */}
      {recyclingGuide.length > 0 && (
        <div className="mt-8 p-6 bg-green-50 rounded-2xl border border-green-200 shadow-md w-full max-w-md">
          <h3 className="text-2xl font-bold text-green-700 mb-4">Guida al Riciclo</h3>
          <ul className="list-disc pl-5 space-y-3 text-gray-700">
            {recyclingGuide.map((item, index) => (
              <li key={index} className="text-lg leading-relaxed">
                <span className="font-semibold text-green-800">{item.label}:</span> {item.tip}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Esponi il componente globalmente in modo che l'HTML possa renderlo
window.WasteRecognizer = WasteRecognizer;
