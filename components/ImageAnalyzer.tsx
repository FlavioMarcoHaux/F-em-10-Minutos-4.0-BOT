
import React, { useState, useRef, useContext } from 'react';
import { analyzeImage } from '../services/geminiService';
import { SpinnerIcon } from './icons';
import { LanguageContext, LanguageContextType } from '../context';
import { usePersistentState, usePersistentBlob } from '../hooks/usePersistentState';


export const ImageAnalyzer: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, , setPreviewBlob] = usePersistentBlob('imageAnalyzer_previewBlob');
  const [analysis, setAnalysis] = usePersistentState('imageAnalyzer_analysis', '');
  const [prompt, setPrompt] = usePersistentState('imageAnalyzer_prompt', '');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { language, t } = useContext(LanguageContext) as LanguageContextType;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setPreviewBlob(file);
      setAnalysis('');
      setError('');
    }
  };

  const handleAnalyze = async () => {
    if (!imageFile && !previewUrl) {
      setError(t('imageErrorSelect'));
      return;
    }
    
    // This is tricky because File object cannot be stored in localStorage.
    // We will rely on user re-selecting the file if the page reloads mid-operation.
    // A more advanced solution would involve storing the image in IndexedDB.
    if (!imageFile) {
        setError(t('imageErrorReselect')); // A new error message might be needed
        return;
    }

    setIsLoading(true);
    setError('');
    setAnalysis('');
    try {
      const result = await analyzeImage(imageFile, prompt, language);
      setAnalysis(result);
    } catch (e) {
      setError(t('imageErrorAnalyze'));
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-lg animate-fade-in">
      <h2 className="text-2xl font-bold text-amber-400 mb-4">{t('imageTitle')}</h2>
      <p className="text-gray-300 mb-4">
        {t('imageDescription')}
      </p>

      <div className="flex flex-col items-center gap-4 mb-4">
        <div 
          className="w-full h-64 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center bg-gray-700 cursor-pointer hover:border-amber-400 transition"
          onClick={() => fileInputRef.current?.click()}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="max-h-full max-w-full object-contain rounded" />
          ) : (
            <p className="text-gray-300">{t('imageUpload')}</p>
          )}
        </div>
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

       <div className="flex flex-col sm:flex-row gap-2 mb-4">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={t('imagePlaceholder')}
          className="flex-grow bg-gray-700 text-white placeholder-gray-400 p-3 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500 transition"
          disabled={isLoading}
        />
        <button
          onClick={handleAnalyze}
          disabled={isLoading || (!imageFile && !previewUrl)}
          className="flex items-center justify-center bg-amber-500 text-gray-900 font-bold py-3 px-6 rounded-lg hover:bg-amber-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {isLoading ? (
            <>
              <SpinnerIcon />
              {t('imageLoading')}
            </>
          ) : (
            t('imageButton')
          )}
        </button>
      </div>

      {error && <p className="text-red-400 mb-4" aria-live="polite">{error}</p>}

      {analysis && (
        <div className="mt-6 p-6 bg-gray-900 rounded-lg border border-gray-700 max-h-96 overflow-y-auto" aria-live="polite">
          <h3 className="text-xl font-semibold text-amber-300 mb-3">{t('imageHeader')}</h3>
          <p className="text-gray-100 whitespace-pre-wrap leading-relaxed">{analysis}</p>
        </div>
      )}
    </div>
  );
};
