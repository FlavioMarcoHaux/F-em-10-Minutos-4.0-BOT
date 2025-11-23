
import React, { useState, useEffect, useContext, useRef } from 'react';
import { 
    generateGuidedPrayer, 
    generateShortPrayer, 
    generateSpeech, 
    generateImageFromPrayer, 
    generateVideo, 
    createThumbnailPromptFromPost,
    generateSocialMediaPost,
    generateYouTubeLongPost,
    MultiSpeakerConfig
} from '../services/geminiService';
import { SpinnerIcon, DownloadIcon } from './icons';
import { LanguageContext, LanguageContextType } from '../context';
import { AspectRatio, SocialMediaPost, YouTubeLongPost, MarketingHistoryItem } from '../types';
import { usePersistentState, usePersistentBlob, idb } from '../hooks/usePersistentState';
import { createOPFSFile, getOPFSFileAsBlob } from '../utils/opfsUtils';

interface MarketingKit {
    prompt: string;
    subthemes: string[];
    prayer: string;
    videoDownloadLink: string | null;
    socialPost: SocialMediaPost | null;
    longPost: YouTubeLongPost | null;
    audioError: string;
    videoError: string;

    imageError: string;
}

const initialMarketingKit: MarketingKit = {
    prompt: '',
    subthemes: ['', '', ''],
    prayer: '',
    videoDownloadLink: null,
    socialPost: null,
    longPost: null,
    audioError: '',
    videoError: '',
    imageError: '',
};

interface GenerationStatus {
    isTextLoading: boolean;
    isAudioLoading: boolean;
    audioProgress: number;
    isVideoLoading: boolean;
    isImageLoading: boolean;
    isFetchingVideo: boolean;
    error: string;
}

const initialGenerationStatus: GenerationStatus = {
    isTextLoading: false,
    isAudioLoading: false,
    audioProgress: 0,
    isVideoLoading: false,
    isImageLoading: false,
    isFetchingVideo: false,
    error: '',
};


const CopyButton = ({ textToCopy }: { textToCopy: string | string[] }) => {
    const { t } = useContext(LanguageContext) as LanguageContextType;
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        const text = Array.isArray(textToCopy) ? textToCopy.join(' ') : textToCopy;
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <button onClick={handleCopy} className="bg-gray-600 text-xs px-2 py-1 rounded hover:bg-gray-500 transition">
            {copied ? t('marketingCopied') : t('marketingCopy')}
        </button>
    );
};

interface MarketingExpertProps {
    history: MarketingHistoryItem[];
    setHistory: React.Dispatch<React.SetStateAction<MarketingHistoryItem[]>>;
}


export const MarketingExpert: React.FC<MarketingExpertProps> = ({ history, setHistory }) => {
    const { language, t } = useContext(LanguageContext) as LanguageContextType;

    const [contentType, setContentType] = usePersistentState<'long' | 'short'>('marketing_contentType', 'long');
    const [longKit, setLongKit] = usePersistentState<MarketingKit>('marketing_longKit', initialMarketingKit);
    const [shortKit, setShortKit] = usePersistentState<MarketingKit>('marketing_shortKit', initialMarketingKit);
    const [longVideoDuration, setLongVideoDuration] = usePersistentState<number>('marketing_longVideoDuration', 10);
    
    // Selectors for current state based on contentType
    const currentKit = contentType === 'long' ? longKit : shortKit;
    const setCurrentKit = contentType === 'long' ? setLongKit : setShortKit;
    
    const [audioObjUrl, audioBlob, setAudioBlob, isAudioLoadingFromDB] = usePersistentBlob(`marketing_${contentType}_audio`);
    const [imageObjUrl, imageBlob, setImageBlob, isImageLoadingFromDB] = usePersistentBlob(`marketing_${contentType}_image`);
    
    const [videoObjUrl, setVideoObjUrl] = useState<string | null>(null);
    const [videoBlob, setVideoBlob] = useState<Blob | null>(null);

    const [generationStatus, setGenerationStatus] = useState<{
        long: GenerationStatus;
        short: GenerationStatus;
    }>({
        long: initialGenerationStatus,
        short: initialGenerationStatus,
    });
    
    const [apiKeySelected, setApiKeySelected] = useState(false);
    
    
    // Derived state for easier access
    const currentStatus = generationStatus[contentType];
    const isAnyMediaGenerating = currentStatus.isAudioLoading || currentStatus.isVideoLoading || currentStatus.isImageLoading || currentStatus.isFetchingVideo;
    const isLongKitBusy = Object.values(generationStatus.long).some(s => typeof s === 'boolean' && s);
    const isShortKitBusy = Object.values(generationStatus.short).some(s => typeof s === 'boolean' && s);

    useEffect(() => {
        const checkKey = async () => {
            if (window.aistudio && typeof window.aistudio.hasSelectedApiKey === 'function') {
                const hasKey = await window.aistudio.hasSelectedApiKey();
                setApiKeySelected(hasKey);
            }
        };
        checkKey();
    }, []);
    
    useEffect(() => {
        let url: string | null = null;
        const fetchVideo = async () => {
            if (currentKit.videoDownloadLink) {
                if (!apiKeySelected) return;
                setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isFetchingVideo: true } }));
                setCurrentKit(prev => ({...prev, videoError: ''}));
                try {
                    const response = await fetch(`${currentKit.videoDownloadLink}&key=${process.env.API_KEY}`);
                    if (!response.ok) { 
                        const errorData = await response.json().catch(() => null);
                        if (errorData?.error?.status === 'NOT_FOUND') {
                           setApiKeySelected(false);
                           throw new Error(t('apiKeyInvalid'));
                        }
                        throw new Error('Failed to fetch video'); 
                    }
                    const blob = await response.blob();
                    setVideoBlob(blob);
                    url = URL.createObjectURL(blob);
                    setVideoObjUrl(url);
                } catch (e: any) {
                    console.error("Error fetching persisted video:", e);
                    setCurrentKit(prev => ({...prev, videoError: e.message || t('videoError')}));
                } finally {
                    setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isFetchingVideo: false } }));
                }
            } else {
                setVideoObjUrl(null);
                setVideoBlob(null);
            }
        };
        fetchVideo();

        return () => {
            if (url) URL.revokeObjectURL(url);
        };
    }, [currentKit.videoDownloadLink, apiKeySelected, contentType, t]);
    
    const handleSubthemeChange = (index: number, value: string) => {
        const newSubthemes = [...currentKit.subthemes];
        newSubthemes[index] = value;
        setCurrentKit(prev => ({...prev, subthemes: newSubthemes }));
    };

    const handleGenerateTextAssets = async () => {
        setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isTextLoading: true, error: '' } }));
        setCurrentKit(prev => ({
            ...prev,
            prayer: '',
            socialPost: null,
            longPost: null,
            videoDownloadLink: null,
            audioError: '',
            videoError: '',
            imageError: '',
        }));
        setAudioBlob(null);
        setImageBlob(null);
        setVideoBlob(null);
        setVideoObjUrl(null);
        
        try {
            const prayerFn = contentType === 'long' ? generateGuidedPrayer : generateShortPrayer;
            
            if (contentType === 'long') {
                const [generatedPrayer, generatedPost] = await Promise.all([
                    prayerFn(currentKit.prompt, language, longVideoDuration),
                    generateYouTubeLongPost(currentKit.prompt, currentKit.subthemes, language, longVideoDuration)
                ]);
                setCurrentKit(prev => ({ ...prev, prayer: generatedPrayer, longPost: generatedPost }));
            } else {
                const generatedPrayer = await prayerFn(currentKit.prompt, language);
                const generatedPost = await generateSocialMediaPost(generatedPrayer, language);
                setCurrentKit(prev => ({ ...prev, prayer: generatedPrayer, socialPost: generatedPost }));
            }

        } catch (e: any) {
            console.error("Text Asset Generation Error:", e);
            let errorMsg = t('prayerError');
            if (e.message?.includes('YouTube long post') || e.message?.includes('social media post')) {
                errorMsg = t('marketingErrorSocial');
            }
            setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], error: errorMsg } }));
        } finally {
            setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isTextLoading: false } }));
        }
    };

    const handleGenerateAudio = async () => {
        if (!currentKit.prayer) return;
        setGenerationStatus(prev => ({ 
            ...prev, 
            [contentType]: { ...prev[contentType], isAudioLoading: true, audioProgress: 0 } 
        }));
        setAudioBlob(null);
        setCurrentKit(prev => ({ ...prev, audioError: '' }));
    
        const multiSpeakerConfig: MultiSpeakerConfig | undefined = contentType === 'long' ? {
            speakers: [
                { name: 'Roberta Erickson', voice: 'Aoede' }, // Female
                { name: 'Milton Dilts', voice: 'Enceladus' }      // Male
            ]
        } : undefined;
    
        // BLADE RUNNER: Use OPFS for direct disk writing to save RAM
        const filename = `marketing_${contentType}_audio_temp.wav`;
        let fileHandle: FileSystemFileHandle | undefined;

        try {
            fileHandle = await createOPFSFile(filename);
        } catch (e) {
            console.warn("OPFS not supported or failed, falling back to memory.", e);
        }

        await generateSpeech(currentKit.prayer, multiSpeakerConfig, {
            onChunk: (pcmData) => {
               // Chunk handled by stream writing if fileHandle exists
            },
            onProgress: (progress) => {
                setGenerationStatus(prev => ({
                    ...prev,
                    [contentType]: { ...prev[contentType], audioProgress: progress }
                }));
            },
            onComplete: async () => {
                if (fileHandle) {
                    try {
                        // Retrieve the full file from disk
                        const finalBlob = await getOPFSFileAsBlob(filename);
                        setAudioBlob(finalBlob);
                    } catch (e) {
                        console.error("Error reading back audio from OPFS", e);
                        setCurrentKit(prev => ({ ...prev, audioError: t('audioError') }));
                    }
                }
                setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isAudioLoading: false } }));
            },
            onError: (errorMsg) => {
                setCurrentKit(prev => ({ ...prev, audioError: t('audioError') }));
                console.error("Audio Generation Error:", errorMsg);
                setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isAudioLoading: false } }));
            }
        }, fileHandle);
    };

    const handleGenerateImage = async () => {
        if (!currentKit.prayer) return;
        setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isImageLoading: true } }));
        setImageBlob(null);
        setCurrentKit(prev => ({ ...prev, imageError: '' }));
        
        try {
            const postContent = currentKit.longPost || currentKit.socialPost;
            if (!postContent) {
                setCurrentKit(prev => ({ ...prev, imageError: t('marketingErrorSocial')}));
                setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isImageLoading: false } }));
                return;
            }
            
            const visualPrompt = await createThumbnailPromptFromPost(
                postContent.title,
                postContent.description,
                currentKit.prayer,
                language
            );
            
            const imageModel = contentType === 'long' ? 'imagen-4.0-generate-001' : 'imagen-4.0-generate-001';
            const aspectRatio: AspectRatio = contentType === 'long' ? '16:9' : '9:16';
            
            const base64Image = await generateImageFromPrayer(visualPrompt, aspectRatio, imageModel);
            const imageResponse = await fetch(`data:image/png;base64,${base64Image}`);
            const blob = await imageResponse.blob();
            setImageBlob(blob);
        } catch (e: any) {
             let errorMsg = t('imageError');
             if (e.message?.includes("thumbnail prompt")) {
                errorMsg = t('mediaPromptError');
            }
            setCurrentKit(prev => ({ ...prev, imageError: errorMsg }));
            console.error("Image Generation Error:", e);
        } finally {
            setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isImageLoading: false } }));
        }
    };
    
     const handleSelectKey = async () => {
        if (window.aistudio && typeof window.aistudio.openSelectKey === 'function') {
            await window.aistudio.openSelectKey();
            setApiKeySelected(true); 
        }
    };

    const handleGenerateVideo = async () => {
        if (!currentKit.prayer) return;
        setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isVideoLoading: true } }));
        setCurrentKit(prev => ({ ...prev, videoDownloadLink: null, videoError: '' }));
        
        try {
            if (!apiKeySelected) { throw new Error("API key not selected."); }
            const postContent = currentKit.longPost || currentKit.socialPost;
            if (!postContent) {
                setCurrentKit(prev => ({ ...prev, videoError: t('marketingErrorSocial') }));
                setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isVideoLoading: false } }));
                return;
            }
            const visualPrompt = await createThumbnailPromptFromPost(
                 postContent.title,
                 postContent.description,
                 currentKit.prayer,
                 language
            );
            const downloadLink = await generateVideo(visualPrompt, contentType === 'long' ? '16:9' : '9:16');
            setCurrentKit(prev => ({ ...prev, videoDownloadLink: downloadLink }));
        } catch (e: any)
        {
            console.error("Video Generation Error:", e);
            const errorPayload = e.error || e;
             if (errorPayload.status === 'NOT_FOUND' || errorPayload.code === 404) {
                setCurrentKit(prev => ({...prev, videoError: t('apiKeyInvalid') }));
                setApiKeySelected(false);
             } else {
                setCurrentKit(prev => ({...prev, videoError: t('videoError') }));
             }
        } finally {
            setGenerationStatus(prev => ({ ...prev, [contentType]: { ...prev[contentType], isVideoLoading: false } }));
        }
    };

    const handleSaveToHistory = async () => {
        const id = Date.now().toString();
        
        const audioBlobKey = audioBlob ? `history_audio_${id}` : undefined;
        const imageBlobKey = imageBlob ? `history_image_${id}` : undefined;
        const videoBlobKey = videoBlob ? `history_video_${id}` : undefined;

        const blobPromises: Promise<void>[] = [];
        if (audioBlob && audioBlobKey) blobPromises.push(idb.set(audioBlobKey, audioBlob));
        if (imageBlob && imageBlobKey) blobPromises.push(idb.set(imageBlobKey, imageBlob));
        if (videoBlob && videoBlobKey) blobPromises.push(idb.set(videoBlobKey, videoBlob));

        try {
            await Promise.all(blobPromises);
            const newHistoryItem: MarketingHistoryItem = {
                id,
                timestamp: Date.now(),
                type: contentType,
                language: language,
                prompt: currentKit.prompt,
                subthemes: currentKit.subthemes,
                prayer: currentKit.prayer,
                socialPost: currentKit.socialPost,
                longPost: currentKit.longPost,
                audioBlobKey,
                imageBlobKey,
                videoBlobKey,
                isDownloaded: false,
            };
            setHistory(prev => [newHistoryItem, ...prev]);
            alert(t('marketingKitSaved'));
        } catch (error) {
            console.error("Failed to save kit to history:", error);
            alert(t('marketingErrorSaving'));
        }
    };

    const isSaveDisabled = !audioBlob || !imageBlob || currentStatus.isTextLoading;

    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg animate-fade-in space-y-6">
            <div>
                <h2 className="text-2xl font-bold text-amber-400 mb-2">{t('marketingTitle')}</h2>
                <p className="text-gray-300">{t('marketingDescription')}</p>
            </div>
            
            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-200 mb-2">{t('marketingSelectType')}</label>
                    <div className="flex gap-2 p-1 bg-gray-700 rounded-lg">
                        <button onClick={() => setContentType('long')} className={`flex-1 py-2 px-4 rounded-md text-sm transition flex items-center justify-center gap-2 ${contentType === 'long' ? 'bg-amber-500 text-gray-900' : 'hover:bg-gray-600 text-gray-200'}`}>
                           {isLongKitBusy && <SpinnerIcon className="animate-spin h-4 w-4" />}
                           {t('marketingLongVideo')}
                        </button>
                        <button onClick={() => setContentType('short')} className={`flex-1 py-2 px-4 rounded-md text-sm transition flex items-center justify-center gap-2 ${contentType === 'short' ? 'bg-amber-500 text-gray-900' : 'hover:bg-gray-600 text-gray-200'}`}>
                           {isShortKitBusy && <SpinnerIcon className="animate-spin h-4 w-4" />}
                           {t('marketingShortVideo')}
                        </button>
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-200 mb-2">{t('marketingDefineTheme')}</label>
                    <input
                        type="text"
                        value={currentKit.prompt}
                        onChange={(e) => setCurrentKit(prev => ({...prev, prompt: e.target.value}))}
                        placeholder={t('prayerPlaceholder')}
                        className="w-full bg-gray-700 text-white placeholder-gray-400 p-3 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 transition"
                        disabled={currentStatus.isTextLoading}
                    />
                </div>
                
                {contentType === 'long' && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-200 mb-2">{t('marketingDefineSubthemes')}</label>
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                                {[0, 1, 2].map(i => (
                                    <input
                                        key={i}
                                        type="text"
                                        value={currentKit.subthemes[i]}
                                        onChange={(e) => handleSubthemeChange(i, e.target.value)}
                                        placeholder={t('marketingSubthemePlaceholder').replace('{number}', String(i + 1))}
                                        className="w-full bg-gray-700 text-white placeholder-gray-400 p-2 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 transition text-sm"
                                        disabled={currentStatus.isTextLoading}
                                    />
                                ))}
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-200 mb-2">{t('prayerDurationLabel')}</label>
                            <select
                                value={longVideoDuration}
                                onChange={(e) => setLongVideoDuration(Number(e.target.value))}
                                className="w-full bg-gray-700 text-white p-3 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 transition"
                                disabled={currentStatus.isTextLoading}
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
                        </div>
                    </div>
                )}


                <button onClick={handleGenerateTextAssets} disabled={currentStatus.isTextLoading || !currentKit.prompt} className="w-full flex items-center justify-center bg-amber-500 text-gray-900 font-bold py-3 px-6 rounded-lg hover:bg-amber-600 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed transform hover:scale-105">
                    {currentStatus.isTextLoading ? <><SpinnerIcon /> {t('marketingGeneratingKit')}</> : t('marketingGenerateKit')}
                </button>
            </div>
            
            {currentStatus.error && <p className="text-red-400 text-center" aria-live="polite">{currentStatus.error}</p>}

            {(currentKit.prayer || currentKit.socialPost || currentKit.longPost || currentStatus.isTextLoading) && (
                <div className="mt-6 p-6 bg-gray-900 rounded-lg border border-gray-700 space-y-6">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xl font-semibold text-amber-300">{t('marketingKitHeader')}</h3>
                        {currentKit.prayer && !currentStatus.isTextLoading && (
                             <div title={isSaveDisabled ? t('marketingSaveDisabledTooltip') : ''}>
                                <button 
                                    onClick={handleSaveToHistory} 
                                    disabled={isSaveDisabled}
                                    className="bg-amber-600 text-white text-xs font-bold py-2 px-3 rounded-lg hover:bg-amber-700 transition disabled:bg-gray-600 disabled:cursor-not-allowed"
                                >
                                    {t('marketingSaveToHistory')}
                                </button>
                            </div>
                        )}
                    </div>

                     {/* Script & Social */}
                    <div className="p-4 bg-gray-800 rounded-lg space-y-4">
                        {currentStatus.isTextLoading ? <div className="flex justify-center"><SpinnerIcon /></div> : (
                            <>
                                <div>
                                    <div className="flex justify-between items-center mb-1">
                                        <h4 className="font-bold text-gray-200">{t('marketingScript')}</h4>
                                        {currentKit.prayer && <CopyButton textToCopy={currentKit.prayer} />}
                                    </div>
                                    <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap text-sm leading-relaxed max-h-40 overflow-y-auto">{currentKit.prayer}</p>
                                </div>
                                <div className="space-y-3 text-sm pt-4 border-t border-gray-700">
                                    <h4 className="font-bold text-gray-200">{t('marketingPostCopy')}</h4>
                                    {currentKit.longPost ? (
                                        <>
                                            <div>
                                                <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTitleLabel')} <CopyButton textToCopy={currentKit.longPost.title} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100">{currentKit.longPost.title}</p>
                                            </div>
                                            <div>
                                                <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingDescriptionLabel')} <CopyButton textToCopy={currentKit.longPost.description} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap">{currentKit.longPost.description}</p>
                                            </div>
                                             <div>
                                                 <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingHashtagsLabel')} <CopyButton textToCopy={currentKit.longPost.hashtags} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-sky-300 text-xs flex flex-wrap gap-x-2 gap-y-1">{currentKit.longPost.hashtags.join(' ')}</p>
                                            </div>
                                            <div>
                                                 <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTimestampsLabel')} <CopyButton textToCopy={currentKit.longPost.timestamps} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap">{currentKit.longPost.timestamps}</p>
                                            </div>
                                            <div>
                                                 <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTagsLabel')} <CopyButton textToCopy={currentKit.longPost.tags} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-sky-300 text-xs flex flex-wrap gap-x-2 gap-y-1">{currentKit.longPost.tags.join(', ')}</p>
                                            </div>
                                        </>
                                    ) : currentKit.socialPost ? (
                                         <>
                                            <div>
                                                <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTitleLabel')} <CopyButton textToCopy={currentKit.socialPost.title} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100">{currentKit.socialPost.title}</p>
                                            </div>
                                            <div>
                                                <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingDescriptionLabel')} <CopyButton textToCopy={currentKit.socialPost.description} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap">{currentKit.socialPost.description}</p>
                                            </div>
                                            <div>
                                                 <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingHashtagsLabel')} <CopyButton textToCopy={currentKit.socialPost.hashtags.map(h => `#${h}`).join(' ')} /></label>
                                                <p className="p-2 bg-gray-700 rounded mt-1 text-sky-300 text-xs flex flex-wrap gap-2">{currentKit.socialPost.hashtags.map(h => `#${h}`).join(' ')}</p>
                                            </div>
                                        </>
                                    ) : (currentKit.prayer && <p className="text-xs text-gray-400">{t('marketingErrorSocial')}</p>)}
                                </div>
                            </>
                        )}
                    </div>

                    {/* Media Generation Section */}
                    {currentKit.prayer && !currentStatus.isTextLoading && (
                         <div className="p-4 bg-gray-800 rounded-lg space-y-4">
                             <h4 className="font-bold text-amber-300 border-b border-gray-700 pb-2 mb-4">{t('marketingMediaGenerationOptions')}</h4>
                             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                 {/* Audio */}
                                <div className="p-4 bg-gray-900 rounded-lg space-y-2 flex flex-col">
                                     <h4 className="font-bold text-gray-200">{t('marketingAudio')}</h4>
                                     <button onClick={handleGenerateAudio} disabled={isAnyMediaGenerating} className="w-full flex items-center justify-center bg-sky-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-700 disabled:bg-gray-600">
                                         {currentStatus.isAudioLoading ? <><SpinnerIcon /> {t('generatingAudio')}</> : t('generateAudio')}
                                     </button>
                                     <div className="flex-grow pt-2">
                                         {currentStatus.isAudioLoading && (
                                            <div className="w-full bg-gray-700 rounded-full h-2.5 my-2">
                                                <div 
                                                    className="bg-sky-500 h-2.5 rounded-full transition-all duration-300" 
                                                    style={{ width: `${currentStatus.audioProgress}%` }}
                                                ></div>
                                            </div>
                                         )}
                                         {isAudioLoadingFromDB && <div className="text-center text-gray-300 italic text-xs">Loading saved audio... <SpinnerIcon/></div>}
                                         {currentKit.audioError && <p className="text-xs text-red-400">{currentKit.audioError}</p>}
                                         {audioObjUrl && !currentStatus.isAudioLoading && (
                                            <div className="flex items-center gap-2">
                                                 <audio controls src={audioObjUrl} className="w-full"></audio>
                                                 <a href={audioObjUrl} download="narration.wav" title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300"><DownloadIcon/></a>
                                            </div>
                                         )}
                                     </div>
                                </div>
                                 {/* Image */}
                                <div className="p-4 bg-gray-900 rounded-lg space-y-2 flex flex-col">
                                    <h4 className="font-bold text-gray-200">{t('generateImage')}</h4>
                                    <button onClick={handleGenerateImage} disabled={isAnyMediaGenerating} className="w-full flex items-center justify-center bg-sky-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-700 disabled:bg-gray-600">
                                        {currentStatus.isImageLoading ? <><SpinnerIcon /> {t('generatingImage')}</> : t('generateImage')}
                                    </button>
                                    <div className="flex-grow pt-2">
                                        <p className="text-xs text-gray-400">{t('imageInfoBilled')}</p>
                                        {isImageLoadingFromDB && <div className="text-center text-gray-300 italic text-xs">Loading saved image... <SpinnerIcon/></div>}
                                        {currentKit.imageError && <p className="text-xs text-red-400">{currentKit.imageError}</p>}
                                        {imageObjUrl && (
                                             <div className="text-center mt-2">
                                                <img src={imageObjUrl} alt="Generated visual" className="rounded-md max-h-40 mx-auto" />
                                                 <a href={imageObjUrl} download="visual.png" title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300 text-xs inline-flex items-center gap-1 mt-1"><DownloadIcon/> {t('downloadMedia')}</a>
                                             </div>
                                        )}
                                    </div>
                                </div>
                                 {/* Video */}
                                <div className="p-4 bg-gray-900 rounded-lg space-y-2 flex flex-col">
                                    <h4 className="font-bold text-gray-200">{t('generateVideo')}</h4>
                                     <button onClick={handleGenerateVideo} disabled={isAnyMediaGenerating || !apiKeySelected} className="w-full flex items-center justify-center bg-sky-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-700 disabled:bg-gray-600">
                                         {currentStatus.isVideoLoading ? <><SpinnerIcon /> {t('generatingVideo')}</> : t('generateVideo')}
                                     </button>
                                     <div className="flex-grow pt-2">
                                          {!apiKeySelected && (
                                            <div className="p-2 bg-yellow-900 border border-yellow-700 rounded-lg text-xs text-yellow-200">
                                                <p>{t('apiKeyInfo')}</p>
                                                <button onClick={handleSelectKey} className="mt-1 font-bold underline hover:text-white">{t('selectApiKey')}</button>
                                            </div>
                                         )}
                                         {currentKit.videoError && <p className="text-xs text-red-400">{currentKit.videoError}</p>}
                                         {currentStatus.isFetchingVideo && <div className="text-center text-gray-300 italic text-xs">Loading saved video... <SpinnerIcon/></div>}
                                         {currentStatus.isVideoLoading && <div className="text-center text-gray-300 italic text-xs">Video generation can take a few minutes... <SpinnerIcon/></div>}
                                         {videoObjUrl && !currentStatus.isFetchingVideo && !currentStatus.isVideoLoading && (
                                             <div className="text-center mt-2">
                                                <video src={videoObjUrl} controls className="rounded-md max-h-40 mx-auto" />
                                                <a href={videoObjUrl} download="visual.mp4" title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300 text-xs inline-flex items-center gap-1 mt-1"><DownloadIcon/> {t('downloadMedia')} Video</a>
                                             </div>
                                         )}
                                     </div>
                                </div>
                             </div>
                         </div>
                    )}


                    {/* Next Steps */}
                    <div className="p-4 bg-teal-900 border border-teal-700 rounded-lg space-y-2">
                        <h4 className="font-bold text-teal-200">{t('marketingNextSteps')}</h4>
                        <ol className="list-decimal list-inside text-sm text-teal-300 space-y-1">
                            <li>{t('marketingStep1')}</li>
                            <li>{t('marketingStep2')}</li>
                            <li>{t('marketingStep3')}</li>
                        </ol>
                        <div className="pt-2">
                            <h5 className="text-xs font-bold text-teal-200">{t('marketingUploadTo')}</h5>
                            <div className="flex gap-4 mt-1">
                                <a href="https://studio.youtube.com/" target="_blank" rel="noopener noreferrer" className="text-sm underline hover:text-white">YouTube</a>
                                <a href="https://www.instagram.com/" target="_blank" rel="noopener noreferrer" className="text-sm underline hover:text-white">Instagram</a>
                                <a href="https://www.tiktok.com/upload" target="_blank" rel="noopener noreferrer" className="text-sm underline hover:text-white">TikTok</a>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
