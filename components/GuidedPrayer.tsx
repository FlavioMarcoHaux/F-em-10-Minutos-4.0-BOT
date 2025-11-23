
import React, { useState, useEffect, useContext, useRef } from 'react';
import { generateSpeech, generateImageFromPrayer, generateVideo, createMediaPromptFromPrayer } from '../services/geminiService';
import { SpinnerIcon, DownloadIcon } from './icons';
import { LanguageContext, LanguageContextType } from '../context';
import { AspectRatio } from '../types';
import { usePersistentState, usePersistentBlob } from '../hooks/usePersistentState';
import { generateGuidedPrayer } from '../services/geminiService';
import { createOPFSFile, getOPFSFileAsBlob } from '../utils/opfsUtils';

interface PrayerGeneratorProps {
    titleKey: string;
    descriptionKey: string;
    prayerGeneratorFn: (prompt: string, language: string, duration?: number) => Promise<string>;
    storageKeyPrefix: string;
    showDurationSelector?: boolean;
}

export const PrayerGenerator: React.FC<PrayerGeneratorProps> = ({ titleKey, descriptionKey, prayerGeneratorFn, storageKeyPrefix, showDurationSelector = false }) => {
    const [prayer, setPrayer] = usePersistentState(`${storageKeyPrefix}_prayer`, '');
    const [prompt, setPrompt] = usePersistentState(`${storageKeyPrefix}_prompt`, '');
    const [duration, setDuration] = usePersistentState<number>(`${storageKeyPrefix}_duration`, 10);
    
    // Persistent Blobs
    const [audioUrl, , setAudioBlob, isAudioLoadingFromDB] = usePersistentBlob(`${storageKeyPrefix}_audio`);
    const [imageUrl, , setImageBlob, isImageLoadingFromDB] = usePersistentBlob(`${storageKeyPrefix}_image`);

    const [videoUrl, setVideoUrl] = usePersistentState<string | null>(`${storageKeyPrefix}_videoUrl`, null);

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const { language, t } = useContext(LanguageContext) as LanguageContextType;

    const [isAudioLoading, setIsAudioLoading] = useState(false);
    const [audioProgress, setAudioProgress] = useState(0);
    const [audioError, setAudioError] = useState('');
    
    const [isVideoLoading, setIsVideoLoading] = useState(false);
    const [videoError, setVideoError] = useState('');

    const [isImageLoading, setIsImageLoading] = useState(false);
    const [imageError, setImageError] = useState('');
    
    const [videoAspectRatio, setVideoAspectRatio] = useState<AspectRatio>('16:9');
    const [imageAspectRatio, setImageAspectRatio] = useState<AspectRatio>('1:1');
    const [apiKeySelected, setApiKeySelected] = useState(false);
    
    const isAnyMediaGenerating = isAudioLoading || isVideoLoading || isImageLoading;

    useEffect(() => {
        const checkKey = async () => {
            if (window.aistudio && typeof window.aistudio.hasSelectedApiKey === 'function') {
                const hasKey = await window.aistudio.hasSelectedApiKey();
                setApiKeySelected(hasKey);
            }
        };
        checkKey();
    }, []);

    const handleRateLimitError = (e: any): boolean => {
      const errorPayload = e.error || e;
      if (errorPayload.status === 'RESOURCE_EXHAUSTED' || errorPayload.code === 429) {
        return true;
      }
      return false;
    }

    const handleGenerate = async (useRandomTheme: boolean = false) => {
        setIsLoading(true);
        setError('');
        setPrayer('');
        // Clear previous media
        setAudioBlob(null);
        setAudioError('');
        setVideoUrl(null);
        setVideoError('');
        setImageBlob(null);
        setImageError('');
        try {
            const result = await prayerGeneratorFn(useRandomTheme ? '' : prompt, language, duration);
            setPrayer(result);
        } catch (e) {
            setError(handleRateLimitError(e) ? t('errorRateLimit') : t('prayerError'));
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleGenerateAudio = async () => {
        if (!prayer) return;
        setIsAudioLoading(true);
        setAudioProgress(0);
        setAudioBlob(null);
        setAudioError('');
    
        const multiSpeakerConfig = storageKeyPrefix === 'guidedPrayer' ? {
            speakers: [
                { name: 'Roberta Erickson', voice: 'Aoede' }, // Female
                { name: 'Milton Dilts', voice: 'Enceladus' }      // Male
            ]
        } : undefined;
        
        // BLADE RUNNER: Use OPFS for direct disk writing to save RAM
        const filename = `${storageKeyPrefix}_audio_temp.wav`;
        let fileHandle: FileSystemFileHandle | undefined;

        try {
            fileHandle = await createOPFSFile(filename);
        } catch (e) {
            console.warn("OPFS not supported or failed, falling back to memory.", e);
        }
    
        await generateSpeech(prayer, multiSpeakerConfig, {
            onChunk: (pcmData) => {
                // Chunk handling for visualization (optional) or fallback if no fileHandle
            },
            onProgress: (progress) => {
                setAudioProgress(progress);
            },
            onComplete: async () => {
                if (fileHandle) {
                    try {
                        // Retrieve the full file from disk
                        const finalBlob = await getOPFSFileAsBlob(filename);
                        setAudioBlob(finalBlob);
                    } catch (e) {
                        console.error("Error reading back audio from OPFS", e);
                        setAudioError(t('audioError'));
                    }
                }
                setIsAudioLoading(false);
            },
            onError: (errorMsg) => {
                setAudioError(t('audioError'));
                console.error("Audio Generation Error:", errorMsg);
                setIsAudioLoading(false);
            }
        }, fileHandle);
    };
    
    const handleGenerateImage = async () => {
        if (!prayer) return;
        setIsImageLoading(true);
        setImageError('');
        setImageBlob(null);
        try {
            const visualPrompt = await createMediaPromptFromPrayer(prayer, language);
            const base64Image = await generateImageFromPrayer(visualPrompt, imageAspectRatio);
            const imageResponse = await fetch(`data:image/png;base64,${base64Image}`);
            const imageBlob = await imageResponse.blob();
            setImageBlob(imageBlob);
        } catch (e: any) {
            if (e.message?.includes("media prompt")) {
                setImageError(t('mediaPromptError'));
            } else if (handleRateLimitError(e)) {
                setImageError(t('errorRateLimit'));
            } else {
                setImageError(t('imageError'));
            }
            console.error("Image Generation Error:", e);
        } finally {
            setIsImageLoading(false);
        }
    };
    
    const handleSelectKey = async () => {
        if (window.aistudio && typeof window.aistudio.openSelectKey === 'function') {
            await window.aistudio.openSelectKey();
            setApiKeySelected(true); 
        }
    };

    const handleGenerateVideo = async () => {
        if (!prayer) return;
        setIsVideoLoading(true);
        setVideoError('');
        setVideoUrl(null);
        
        try {
            if (!apiKeySelected) {
                throw new Error("API key not selected.");
            }
            const visualPrompt = await createMediaPromptFromPrayer(prayer, language);
            const downloadLink = await generateVideo(visualPrompt, videoAspectRatio);
            const response = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
            if (!response.ok) {
                let errorData = { message: `Failed to fetch video: ${response.statusText}` };
                try {
                    const resJson = await response.json();
                    if (resJson.error) {
                       errorData = { ...errorData, ...resJson.error };
                    }
                } catch (jsonError) {
                    // Ignore if response is not JSON
                }
                 throw errorData;
            }
            const videoBlob = await response.blob();
            setVideoUrl(URL.createObjectURL(videoBlob));

        } catch (e: any) {
            console.error("Video Generation Error:", e);

            let finalError = t('videoError');
            const errorPayload = e.error || e;

            if (typeof e.message === 'string' && e.message.includes("media prompt")) {
                finalError = t('mediaPromptError');
            } else if (errorPayload.status === 'RESOURCE_EXHAUSTED' || errorPayload.code === 429) {
                finalError = t('errorRateLimit');
            } else if (errorPayload.status === 'NOT_FOUND' || errorPayload.code === 404) {
                finalError = t('apiKeyInvalid');
                setApiKeySelected(false);
            }
            
            setVideoError(finalError);
        } finally {
            setIsVideoLoading(false);
        }
    };

    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg animate-fade-in space-y-6">
            <div>
                <h2 className="text-2xl font-bold text-amber-400 mb-2">{t(titleKey)}</h2>
                <p className="text-gray-300">{t(descriptionKey)}</p>
            </div>

            <div className="space-y-4">
                 <button
                    onClick={() => handleGenerate(true)}
                    disabled={isLoading}
                    className="w-full flex items-center justify-center bg-teal-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-teal-700 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed transform hover:scale-105"
                >
                    {isLoading ? (<><SpinnerIcon />{t('prayerLoading')}</>) : (t('prayerChoseYouButton'))}
                </button>

                <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-600"></div>
                    </div>
                    <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-gray-800 text-gray-400">{t('orText')}</span>
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-200 mb-2">{t('prayerDefineTheme')}</label>
                    <div className="flex flex-col sm:flex-row gap-2">
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder={t('prayerPlaceholder')}
                            className="flex-grow bg-gray-700 text-white placeholder-gray-400 p-3 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 transition"
                            disabled={isLoading}
                        />
                         {showDurationSelector && (
                             <select
                                value={duration}
                                onChange={(e) => setDuration(Number(e.target.value))}
                                className="bg-gray-700 text-white p-3 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 transition"
                                disabled={isLoading}
                                aria-label={t('prayerDurationLabel')}
                            >
                                <option value={5}>{t('prayerDuration5min')}</option>
                                <option value={10}>{t('prayerDuration10min')}</option>
                                <option value={15}>{t('prayerDuration15min')}</option>
                                <option value={20}>{t('prayerDuration20min')}</option>
                                <option value={30}>{t('prayerDuration30min')}</option>
                                <option value={45}>{t('prayerDuration45min')}</option>
                                <option value={60}>{t('prayerDuration60min')}</option>
                            </select>
                        )}
                        <button
                            onClick={() => handleGenerate(false)}
                            disabled={isLoading || !prompt}
                            className="flex items-center justify-center bg-amber-500 text-gray-900 font-bold py-3 px-6 rounded-lg hover:bg-amber-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed transform hover:scale-105"
                        >
                            {isLoading ? (<><SpinnerIcon />{t('prayerLoading')}</>) : (t('prayerButton'))}
                        </button>
                    </div>
                </div>
            </div>

            {error && <p className="text-red-400 text-center" aria-live="polite">{error}</p>}

            {prayer && (
                <div className="mt-6 p-6 bg-gray-900 rounded-lg border border-gray-700" aria-live="polite">
                    <h3 className="text-xl font-semibold text-amber-300 mb-3">{t('prayerHeader')}</h3>
                    <p className="text-gray-100 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">{prayer}</p>
                </div>
            )}

            {prayer && !isLoading && (
                <div className="mt-6 p-4 bg-gray-700 rounded-lg space-y-6">
                    <h3 className="text-xl font-bold text-amber-300 border-b border-gray-600 pb-2">{t('generateOptions')}</h3>
                    
                    {/* Audio Generation */}
                    <div className="space-y-2">
                       <button onClick={handleGenerateAudio} disabled={isAnyMediaGenerating} className="flex items-center justify-center bg-sky-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed">
                           {isAudioLoading ? <><SpinnerIcon /> {t('generatingAudio')}</> : t('generateAudio')}
                       </button>
                       {isAudioLoading && (
                            <div className="w-full bg-gray-600 rounded-full h-2.5 my-2">
                                <div 
                                    className="bg-sky-500 h-2.5 rounded-full transition-all duration-300" 
                                    style={{ width: `${audioProgress}%` }}
                                ></div>
                            </div>
                       )}
                       {audioError && <p className="text-red-400 text-sm mt-2">{audioError}</p>}
                       {isAudioLoadingFromDB && <div className="text-center text-gray-300 italic text-xs">Loading saved audio... <SpinnerIcon/></div>}
                       {audioUrl && !isAudioLoading && (
                            <div className="flex items-center gap-4 mt-2">
                                <audio controls src={audioUrl} className="w-full max-w-sm"></audio>
                                <a href={audioUrl} download={`${storageKeyPrefix}_audio.wav`} title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300 transition-colors">
                                    <DownloadIcon />
                                </a>
                            </div>
                       )}
                       <p className="text-xs text-gray-400 pl-1">{t('audioInfo')}</p>
                    </div>

                    {/* Image Generation */}
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center gap-4">
                            <label htmlFor={`image-aspect-ratio-${storageKeyPrefix}`} className="font-semibold text-gray-200">{t('aspectRatio')}:</label>
                            <select
                                id={`image-aspect-ratio-${storageKeyPrefix}`}
                                value={imageAspectRatio}
                                onChange={(e) => setImageAspectRatio(e.target.value as AspectRatio)}
                                className="bg-gray-800 text-white p-2 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-sky-500 text-sm"
                                disabled={isAnyMediaGenerating}
                            >
                                <option value="1:1">1:1 (Square)</option>
                                <option value="16:9">16:9 (Landscape)</option>
                                <option value="9:16">9:16 (Portrait)</option>
                                <option value="4:3">4:3 (Standard)</option>
                                <option value="3:4">3:4 (Tall)</option>
                            </select>
                        </div>
                       <button onClick={handleGenerateImage} disabled={isAnyMediaGenerating} className="flex items-center justify-center bg-sky-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed">
                           {isImageLoading ? <><SpinnerIcon /> {t('generatingImage')}</> : t('generateImage')}
                       </button>
                       <p className="text-xs text-gray-400 pl-1">{t('imageInfoBilled')}</p>
                       {imageError && <p className="text-red-400 text-sm">{imageError}</p>}
                        {isImageLoading && <div className="flex justify-center"><SpinnerIcon /></div>}
                        {isImageLoadingFromDB && <div className="text-center text-gray-300 italic text-xs">Loading saved image... <SpinnerIcon/></div>}
                        {imageUrl && (
                            <div className="mt-4 text-center">
                                <img src={imageUrl} alt="Generated art" className="rounded-lg max-w-sm mx-auto shadow-lg" />
                                <a href={imageUrl} download={`${storageKeyPrefix}_art.png`} title={t('downloadMedia')} className="inline-flex items-center gap-2 mt-3 text-sky-400 hover:text-sky-300 transition-colors">
                                    <DownloadIcon />
                                    <span>{t('downloadMedia')}</span>
                                </a>
                            </div>
                        )}
                    </div>

                    {/* Video Generation */}
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center gap-4">
                            <label htmlFor={`video-aspect-ratio-${storageKeyPrefix}`} className="font-semibold text-gray-200">{t('aspectRatio')}:</label>
                            <select
                                id={`video-aspect-ratio-${storageKeyPrefix}`}
                                value={videoAspectRatio}
                                onChange={(e) => setVideoAspectRatio(e.target.value as AspectRatio)}
                                className="bg-gray-800 text-white p-2 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-sky-500 text-sm"
                                disabled={isAnyMediaGenerating}
                            >
                                <option value="16:9">16:9 (Landscape)</option>
                                <option value="9:16">9:16 (Portrait)</option>
                            </select>
                        </div>
                       <button onClick={handleGenerateVideo} disabled={isAnyMediaGenerating || !apiKeySelected} className="flex items-center justify-center bg-sky-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed">
                          {isVideoLoading ? <><SpinnerIcon /> {t('generatingVideo')}</> : t('generateVideo')}
                       </button>
                        {!apiKeySelected && (
                            <div className="p-3 bg-yellow-900 border border-yellow-700 rounded-lg text-sm text-yellow-200">
                                <p>{t('apiKeyInfo')}</p>
                                <button onClick={handleSelectKey} className="mt-2 font-bold underline hover:text-white">{t('selectApiKey')}</button>
                                <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="ml-4 text-xs underline hover:text-white">{t('apiKeyLink')}</a>
                            </div>
                        )}
                        <p className="text-xs text-gray-400 pl-1">{t('videoRateLimitWarning')}</p>
                       {videoError && <p className="text-red-400 text-sm">{videoError}</p>}
                       {isVideoLoading && <div className="text-center text-gray-300 italic">Video generation can take a few minutes... <SpinnerIcon/></div>}
                       {videoUrl && (
                           <div className="mt-4 text-center">
                                <video src={videoUrl} controls className="rounded-lg max-w-sm mx-auto shadow-lg" />
                                <a href={videoUrl} download={`${storageKeyPrefix}_video.mp4`} title={t('downloadMedia')} className="inline-flex items-center gap-2 mt-3 text-sky-400 hover:text-sky-300 transition-colors">
                                    <DownloadIcon />
                                    <span>{t('downloadMedia')}</span>
                                </a>
                           </div>
                       )}
                    </div>
                </div>
            )}
        </div>
    );
};

export const GuidedPrayer: React.FC = () => {
    return (
        <PrayerGenerator
            titleKey="prayerTitle"
            descriptionKey="prayerDescription"
            prayerGeneratorFn={generateGuidedPrayer}
            storageKeyPrefix="guidedPrayer"
            showDurationSelector={true}
        />
    );
};
