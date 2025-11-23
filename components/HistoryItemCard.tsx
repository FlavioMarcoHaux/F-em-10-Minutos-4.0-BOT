
import React, { useState, useEffect, useContext, Dispatch, SetStateAction } from 'react';
import { MarketingHistoryItem } from '../types';
import { LanguageContext, LanguageContextType } from '../context';
import { SpinnerIcon, DownloadIcon, TrashIcon, CheckIcon } from './icons';
import { idb } from '../hooks/usePersistentState';

// These are loaded from CDN in index.html
declare const JSZip: any;
declare const saveAs: any;


const useBlobLoader = (key?: string) => {
    const [url, setUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (!key) {
            setUrl(null);
            return;
        }
        let isActive = true;
        let objectUrl: string | null = null;
        
        setIsLoading(true);
        idb.get<Blob>(key).then(blob => {
            if (isActive && blob) {
                objectUrl = URL.createObjectURL(blob);
                setUrl(objectUrl);
            }
        }).finally(() => {
            if(isActive) setIsLoading(false)
        });

        return () => {
            isActive = false;
            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
            }
        };
    }, [key]);

    return { url, isLoading };
};

const CopyButton = ({ textToCopy }: { textToCopy: string | string[] }) => {
    const { t } = useContext(LanguageContext) as LanguageContextType;
    const [copied, setCopied] = useState(false);

    const handleCopy = (e: React.MouseEvent) => {
        e.stopPropagation();
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


interface HistoryItemCardProps {
    item: MarketingHistoryItem;
    onDelete: (id: string) => void;
    setHistory: Dispatch<SetStateAction<MarketingHistoryItem[]>>;
}

export const HistoryItemCard: React.FC<HistoryItemCardProps> = ({ item, onDelete, setHistory }) => {
    const { t } = useContext(LanguageContext) as LanguageContextType;
    const [isOpen, setIsOpen] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);

    const { url: audioUrl, isLoading: isAudioLoading } = useBlobLoader(item.audioBlobKey);
    const { url: imageUrl, isLoading: isImageLoading } = useBlobLoader(item.imageBlobKey);
    const { url: videoUrl, isLoading: isVideoLoading } = useBlobLoader(item.videoBlobKey);

    const title = item.longPost?.title || item.socialPost?.title || item.prompt;
    const date = new Date(item.timestamp).toLocaleString(t('appLocaleCode') || 'en-US');

    const handleDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        // The confirmation is now handled by the parent component to ensure correct execution context.
        onDelete(item.id);
    };

    const handleDownloadKit = async (e: React.MouseEvent) => {
        e.stopPropagation();
        setIsDownloading(true);
        try {
            const zip = new JSZip();

            // 1. Create content.txt
            let textContent = `PROMPT: ${item.prompt}\n\n`;
            textContent += `====================\nSCRIPT (PRAYER)\n====================\n\n${item.prayer}\n\n`;
            
            if (item.longPost) {
                textContent += `====================\nYOUTUBE POST\n====================\n\n`;
                textContent += `TITLE: ${item.longPost.title}\n\n`;
                textContent += `DESCRIPTION:\n${item.longPost.description}\n\n`;
                textContent += `HASHTAGS: ${item.longPost.hashtags.join(' ')}\n\n`;
                textContent += `TIMESTAMPS:\n${item.longPost.timestamps}\n\n`;
                textContent += `TAGS: ${item.longPost.tags.join(', ')}\n`;
            } else if (item.socialPost) {
                textContent += `====================\nSOCIAL MEDIA POST\n====================\n\n`;
                textContent += `TITLE: ${item.socialPost.title}\n\n`;
                textContent += `DESCRIPTION:\n${item.socialPost.description}\n\n`;
                textContent += `HASHTAGS: ${item.socialPost.hashtags.map(h => h.startsWith('#') ? h : `#${h}`).join(' ')}\n`;
            }

            zip.file("content.txt", textContent);
            
            // 2. Add media files
            const blobPromises = [
                item.audioBlobKey ? idb.get<Blob>(item.audioBlobKey).then(blob => ({ name: 'narration.wav', blob })) : Promise.resolve(null),
                item.imageBlobKey ? idb.get<Blob>(item.imageBlobKey).then(blob => ({ name: 'visual.png', blob })) : Promise.resolve(null),
                item.videoBlobKey ? idb.get<Blob>(item.videoBlobKey).then(blob => ({ name: 'video.mp4', blob })) : Promise.resolve(null)
            ];

            const mediaFiles = await Promise.all(blobPromises);
            mediaFiles.forEach(file => {
                if (file && file.blob) {
                    zip.file(file.name, file.blob);
                }
            });

            // 3. Generate and download zip with dynamic name
            const zipBlob = await zip.generateAsync({ type: "blob" });
            
            const safeTitle = (title.replace(/[^a-zA-Z0-9]/g, '_') || "kit").substring(0, 50);
            const ts = new Date(item.timestamp);
            const formattedDate = `${String(ts.getDate()).padStart(2, '0')}${String(ts.getMonth() + 1).padStart(2, '0')}${ts.getFullYear()}_${String(ts.getHours()).padStart(2, '0')}${String(ts.getMinutes()).padStart(2, '0')}`;
            const filename = `${safeTitle}_${formattedDate}.zip`;
            
            saveAs(zipBlob, filename);

            // 4. Update download status
            setHistory(prev => prev.map(histItem => 
                histItem.id === item.id ? { ...histItem, isDownloaded: true } : histItem
            ));

        } catch (error) {
            console.error("Failed to create zip file", error);
            alert(t('marketingDownloadError'));
        } finally {
            setIsDownloading(false);
        }
    };

    const post = item.longPost || item.socialPost;

    return (
        <div className="bg-gray-800 rounded-lg border border-gray-700">
            <div className="flex justify-between items-center p-4">
                <div className="flex-1 min-w-0 cursor-pointer hover:opacity-80 transition" onClick={() => setIsOpen(!isOpen)}>
                    <p className="font-bold text-amber-300 truncate pr-2">{title}</p>
                    <div className="flex items-center gap-4">
                        <p className="text-xs text-gray-300">{date}</p>
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-300">{t('historyItemLanguage')}</span>
                            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                                item.language === 'pt' ? 'bg-green-800 text-green-200' :
                                item.language === 'en' ? 'bg-blue-800 text-blue-200' :
                                'bg-yellow-800 text-yellow-200'
                            }`}>
                                {item.language.toUpperCase()}
                            </span>
                             {item.isDownloaded && (
                                <span title={t('historyDownloaded')} className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-600 text-gray-300">
                                    <CheckIcon className="h-3 w-3" />
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                    <button 
                        onClick={handleDownloadKit} 
                        disabled={isDownloading}
                        title={t('marketingDownloadKit')}
                        className="flex items-center justify-center bg-green-600 text-white font-bold p-2 rounded-lg hover:bg-green-700 transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed"
                    >
                        {isDownloading ? (
                            <SpinnerIcon className="animate-spin h-4 w-4 text-white" />
                        ) : (
                            <DownloadIcon className="h-4 w-4" />
                        )}
                    </button>
                    <button 
                        onClick={handleDelete} 
                        title={t('marketingDeleteItem')}
                        className="p-2 text-red-400 rounded-lg hover:bg-red-500/20 hover:text-red-300 transition-colors"
                    >
                        <TrashIcon className="h-4 w-4" />
                    </button>
                    <button onClick={() => setIsOpen(!isOpen)} className="p-2 text-gray-400 rounded-full hover:bg-gray-700">
                        <svg className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>

            {isOpen && (
                 <div className="p-4 border-t border-gray-700 space-y-6 animate-fade-in">
                    {/* Media Section */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="p-2 bg-gray-900 rounded-lg">
                            <h4 className="text-sm font-bold text-gray-200 mb-2 text-center">{t('marketingAudio')}</h4>
                            {isAudioLoading ? <SpinnerIcon/> : audioUrl ? (
                                <div className="flex items-center gap-2">
                                    <audio controls src={audioUrl} className="w-full"></audio>
                                    <a href={audioUrl} download={`audio_${item.id}.wav`} title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300"><DownloadIcon/></a>
                                </div>
                            ) : <p className="text-xs text-gray-500 text-center">N/A</p>}
                        </div>
                         <div className="p-2 bg-gray-900 rounded-lg">
                            <h4 className="text-sm font-bold text-gray-200 mb-2 text-center">{t('generateImage')}</h4>
                            {isImageLoading ? <SpinnerIcon/> : imageUrl ? (
                                <div className="text-center">
                                    <img src={imageUrl} alt="Generated visual" className="rounded-md max-h-40 mx-auto" />
                                    <a href={imageUrl} download={`image_${item.id}.png`} title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300 text-xs inline-flex items-center gap-1 mt-1"><DownloadIcon/> {t('downloadMedia')}</a>
                                </div>
                            ) : <p className="text-xs text-gray-500 text-center">N/A</p>}
                        </div>
                         <div className="p-2 bg-gray-900 rounded-lg">
                            <h4 className="text-sm font-bold text-gray-200 mb-2 text-center">{t('generateVideo')}</h4>
                            {isVideoLoading ? <SpinnerIcon/> : videoUrl ? (
                                <div className="text-center">
                                    <video src={videoUrl} controls className="rounded-md max-h-40 mx-auto" />
                                    <a href={videoUrl} download={`video_${item.id}.mp4`} title={t('downloadMedia')} className="text-sky-400 hover:text-sky-300 text-xs inline-flex items-center gap-1 mt-1"><DownloadIcon/> {t('downloadMedia')}</a>
                                </div>
                            ) : <p className="text-xs text-gray-500 text-center">N/A</p>}
                        </div>
                    </div>
                    {/* Text Section */}
                    <div className="space-y-3 text-sm pt-4 border-t border-gray-700">
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <h4 className="font-bold text-gray-200">{t('marketingScript')}</h4>
                                <CopyButton textToCopy={item.prayer} />
                            </div>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap text-sm leading-relaxed max-h-40 overflow-y-auto">{item.prayer}</p>
                        </div>

                         <h4 className="font-bold text-gray-200 pt-2">{t('marketingPostCopy')}</h4>
                         {post && <div>
                            <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTitleLabel')} <CopyButton textToCopy={post.title} /></label>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100">{post.title}</p>
                        </div>}
                         {post && <div>
                            <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingDescriptionLabel')} <CopyButton textToCopy={post.description} /></label>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap">{post.description}</p>
                        </div>}
                        {post && <div>
                             <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingHashtagsLabel')} <CopyButton textToCopy={post.hashtags.map(h => h.startsWith('#') ? h : `#${h}`)} /></label>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-sky-300 text-xs flex flex-wrap gap-x-2 gap-y-1">{post.hashtags.map(h => h.startsWith('#') ? h : `#${h}`).join(' ')}</p>
                        </div>}
                        
                        {item.longPost && <div>
                             <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTimestampsLabel')} <CopyButton textToCopy={item.longPost.timestamps} /></label>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-gray-100 whitespace-pre-wrap">{item.longPost.timestamps}</p>
                        </div>}
                        {item.longPost && <div>
                             <label className="text-xs font-semibold text-gray-300 flex justify-between items-center">{t('marketingTagsLabel')} <CopyButton textToCopy={item.longPost.tags} /></label>
                            <p className="p-2 bg-gray-700 rounded mt-1 text-sky-300 text-xs flex flex-wrap gap-x-2 gap-y-1">{item.longPost.tags.join(', ')}</p>
                        </div>}
                    </div>
                </div>
            )}
        </div>
    );
};
