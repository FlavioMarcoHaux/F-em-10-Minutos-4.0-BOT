
import React, { useState, useEffect, useContext, useCallback } from 'react';
import {
    getTrendingTopic,
    generateGuidedPrayer,
    generateShortPrayer,
    generateSpeech,
    generateImageFromPrayer,
    createThumbnailPromptFromPost,
    generateSocialMediaPost,
    generateYouTubeLongPost,
    MultiSpeakerConfig
} from '../services/geminiService';
import { SpinnerIcon, BotIcon, YouTubeIcon, TikTokIcon } from './icons';
import { LanguageContext, LanguageContextType } from '../context';
import { AspectRatio, SocialMediaPost, YouTubeLongPost, MarketingHistoryItem } from '../types';
import { usePersistentState, idb } from '../hooks/usePersistentState';
import { createWavFile } from '../utils/audio';

interface BotAgentProps {
    history: MarketingHistoryItem[];
    setHistory: React.Dispatch<React.SetStateAction<MarketingHistoryItem[]>>;
}

export const BotAgent: React.FC<BotAgentProps> = ({ history, setHistory }) => {
    const { t } = useContext(LanguageContext) as LanguageContextType;
    
    // Autonomous Agent States
    const [isAgentLongActive, setIsAgentLongActive] = usePersistentState<boolean>('agent_isLongActive', false);
    const [isAgentShortActive, setIsAgentShortActive] = usePersistentState<boolean>('agent_isShortActive', false);

    const [longVideoCadence, setLongVideoCadence] = usePersistentState<number>('agent_longVideoCadence', 1);
    const [shortVideoCadence, setShortVideoCadence] = usePersistentState<number>('agent_shortVideoCadence', 1);
    
    // New: Duration Selector for Automated Long Videos
    const [agentLongDuration, setAgentLongDuration] = usePersistentState<number>('agent_longDuration', 10);
    
    const [agentStatusLong, setAgentStatusLong] = useState<string>('');
    const [agentStatusShort, setAgentStatusShort] = useState<string>('');

    const [lastRuns, setLastRuns] = usePersistentState<{ [key: string]: number }>('agent_lastRuns', {});
    const [jobsInProgress, setJobsInProgress] = useState<string[]>([]);

    // Integration States
    const [isYoutubeConnected, setIsYoutubeConnected] = usePersistentState('agent_youtubeConnected', false);
    const [isTiktokConnected, setIsTiktokConnected] = usePersistentState('agent_tiktokConnected', false);
    const [connectingPlatform, setConnectingPlatform] = useState<string | null>(null);

    const handleConnect = (platform: 'youtube' | 'tiktok') => {
        setConnectingPlatform(platform);
        setTimeout(() => {
            if (platform === 'youtube') setIsYoutubeConnected(true);
            else setIsTiktokConnected(true);
            setConnectingPlatform(null);
        }, 2000);
    };

    const handleDisconnect = (platform: 'youtube' | 'tiktok') => {
         if (platform === 'youtube') setIsYoutubeConnected(false);
         else setIsTiktokConnected(false);
    };

    const generateAndSaveKit = useCallback(async (
        jobLang: string,
        jobType: 'long' | 'short',
        theme: string,
        subthemes: string[],
    ) => {
        // 1. Generate Text Assets
        let prayer: string;
        let post: SocialMediaPost | YouTubeLongPost;
    
        // Use the configured duration for long videos, or default to 10 if not set
        const duration = jobType === 'long' ? agentLongDuration : 1; 

        if (jobType === 'long') {
            const [p, ps] = await Promise.all([
                generateGuidedPrayer(theme, jobLang, duration),
                generateYouTubeLongPost(theme, subthemes, jobLang, duration)
            ]);
            prayer = p;
            post = ps;
        } else {
            prayer = await generateShortPrayer(theme, jobLang);
            post = await generateSocialMediaPost(prayer, jobLang);
        }
        if (!prayer || !post) throw new Error(`Failed to generate text assets for ${jobLang}.`);
    
        // 2. Generate Media Assets in Parallel
        const audioBlobPromise: Promise<Blob> = new Promise((resolve, reject) => {
            const pcmChunks: Uint8Array[] = [];
            const multiConfig = jobType === 'long' 
                ? { speakers: [{ name: 'Roberta Erickson', voice: 'Aoede' }, { name: 'Milton Dilts', voice: 'Enceladus' }] } 
                : undefined;
            
            generateSpeech(prayer, multiConfig, {
                onChunk: (pcmData) => {
                    pcmChunks.push(pcmData);
                },
                onProgress: () => {}, 
                onComplete: () => {
                    const totalLength = pcmChunks.reduce((acc, chunk) => acc + chunk.length, 0);
                    if (totalLength === 0) {
                        return reject(new Error("Audio generation resulted in empty audio."));
                    }
                    const concatenatedPcm = new Uint8Array(totalLength);
                    let offset = 0;
                    pcmChunks.forEach(chunk => {
                        concatenatedPcm.set(chunk, offset);
                        offset += chunk.length;
                    });
                    const wavBlob = createWavFile(concatenatedPcm, 1, 24000, 16);
                    resolve(wavBlob);
                },
                onError: (errorMsg) => {
                    reject(new Error(errorMsg));
                }
            });
        });
    
        const imageBlobPromise = (async (): Promise<Blob> => {
            const visualPrompt = await createThumbnailPromptFromPost(post.title, post.description, prayer, jobLang);
            const aspectRatio: AspectRatio = jobType === 'short' ? '9:16' : '16:9';
            const imageB64 = await generateImageFromPrayer(visualPrompt, aspectRatio, 'imagen-4.0-generate-001');
            if (!imageB64) throw new Error("Image generation failed");
            const imageResponse = await fetch(`data:image/png;base64,${imageB64}`);
            return await imageResponse.blob();
        })();
    
        const [audioBlob, imageBlob] = await Promise.all([audioBlobPromise, imageBlobPromise]);
    
        if (!audioBlob || !imageBlob) throw new Error("Failed to create media blobs.");
    
        // 3. Save to History
        const id = `${Date.now()}-${jobLang}-${jobType}`;
        const audioBlobKey = `history_audio_${id}`;
        const imageBlobKey = `history_image_${id}`;
        await Promise.all([
            idb.set(audioBlobKey, audioBlob),
            idb.set(imageBlobKey, imageBlob),
        ]);
    
        const newHistoryItem: MarketingHistoryItem = {
            id,
            timestamp: Date.now(),
            type: jobType,
            language: jobLang,
            prompt: theme,
            subthemes: subthemes,
            prayer: prayer,
            socialPost: jobType === 'short' ? post as SocialMediaPost : null,
            longPost: jobType === 'long' ? post as YouTubeLongPost : null,
            audioBlobKey,
            imageBlobKey,
            isDownloaded: false,
        };
        setHistory(prev => [newHistoryItem, ...prev].sort((a, b) => b.timestamp - a.timestamp));
    }, [setHistory, agentLongDuration]);

    const runAutomatedLongVideoBatch = useCallback(async () => {
        const jobIdentifiers = ['pt-long', 'en-long', 'es-long'];
        setJobsInProgress(prev => [...prev, ...jobIdentifiers]);
        try {
            const { theme, subthemes } = await getTrendingTopic('pt', 'long');
            for (const lang of ['pt', 'en', 'es'] as const) {
                await generateAndSaveKit(lang, 'long', theme, subthemes);
            }
        } catch (error) {
            console.error(`Autonomous agent long video batch failed:`, error);
        } finally {
            setJobsInProgress(prev => prev.filter(job => !jobIdentifiers.includes(job)));
        }
    }, [generateAndSaveKit]);

    const runAutomatedShortVideoJob = useCallback(async (jobLang: string) => {
        const jobIdentifier = `${jobLang}-short`;
        setJobsInProgress(prev => [...prev, jobIdentifier]);
        try {
            const { theme, subthemes } = await getTrendingTopic(jobLang, 'short');
            await generateAndSaveKit(jobLang, 'short', theme, subthemes);
        } catch (error) {
             console.error(`Autonomous agent job failed for ${jobLang}/short:`, error);
        } finally {
             setJobsInProgress(prev => prev.filter(job => job !== jobIdentifier));
        }
    }, [generateAndSaveKit]);


    useEffect(() => {
        const schedules = {
            pt: { long: [6, 12, 18], short: [9, 12, 18] },
            en: { long: [7, 13, 19], short: [9, 12, 18] },
            es: { long: [8, 14, 20], short: [9, 12, 18] },
        };
        const offsets = { pt: 0, en: 20, es: 40 };
        
        const findNextLongBatchJob = (cadence: number) => {
            const now = new Date();
            let closestJob = { time: Infinity, details: '' };

            for (let d = 0; d < 2; d++) {
                const checkDate = new Date(now);
                checkDate.setDate(now.getDate() + d);
                const todayStr = checkDate.toISOString().split('T')[0];
                const scheduleHours = schedules['pt'].long.slice(0, cadence);
                
                for (const hour of scheduleHours) {
                    const minute = offsets['pt'];
                    const jobKey = `${todayStr}_pt_long_batch_${hour}:${minute}`;
                    const jobTime = new Date(checkDate);
                    jobTime.setHours(hour, minute, 0, 0);

                    if (jobTime.getTime() > now.getTime() && jobTime.getTime() < closestJob.time && !lastRuns[jobKey]) {
                       closestJob = {
                           time: jobTime.getTime(),
                           details: t('agentStatusIdle')
                             .replace('{type}', t('agentTitleLong'))
                             .replace('{lang}', 'PT, EN, ES')
                             .replace('{time}', jobTime.toLocaleTimeString(t('appLocaleCode'), { hour: '2-digit', minute: '2-digit' }))
                       };
                    }
                }
                if (closestJob.time !== Infinity) break; 
            }
             return closestJob.details;
        };

        const findNextShortJob = (cadence: number) => {
            const now = new Date();
            let closestJob = { time: Infinity, details: '' };

             for (const lang of ['pt', 'en', 'es'] as const) {
                for (let d = 0; d < 2; d++) {
                    const checkDate = new Date(now);
                    checkDate.setDate(now.getDate() + d);
                    const todayStr = checkDate.toISOString().split('T')[0];
                    const scheduleHours = schedules[lang].short.slice(0, cadence);
                    
                    for (const hour of scheduleHours) {
                        const minute = offsets[lang];
                        const jobKey = `${todayStr}_${lang}_short_${hour}:${minute}`;
                        const jobTime = new Date(checkDate);
                        jobTime.setHours(hour, minute, 0, 0);

                        if (jobTime.getTime() > now.getTime() && jobTime.getTime() < closestJob.time && !lastRuns[jobKey]) {
                           closestJob = {
                               time: jobTime.getTime(),
                               details: t('agentStatusIdle')
                                 .replace('{type}', t('marketingShortVideo'))
                                 .replace('{lang}', lang.toUpperCase())
                                 .replace('{time}', jobTime.toLocaleTimeString(t('appLocaleCode'), { hour: '2-digit', minute: '2-digit' }))
                           };
                        }
                    }
                }
             }
             return closestJob.details;
        };

        const updateStatuses = () => {
            const isLongJobRunning = jobsInProgress.some(job => job.endsWith('-long'));
            if (isLongJobRunning) {
                const typeStr = t('marketingLongVideo');
                setAgentStatusLong(t('agentStatusRunning').replace('{type}', typeStr).replace('{lang}', 'PT, EN, ES'));
            } else if (isAgentLongActive) {
                const nextJob = findNextLongBatchJob(longVideoCadence);
                setAgentStatusLong(nextJob || t('agentStatusIdle').replace('{type}','...').replace('{lang}','...').replace('{time}','...'));
            } else {
                setAgentStatusLong(t('agentStatusDisabled'));
            }

            const runningShortJobs = jobsInProgress
                .filter(job => job.endsWith('-short'))
                .map(job => job.split('-')[0].toUpperCase());
            if (runningShortJobs.length > 0) {
                 const typeStr = t('marketingShortVideo');
                 setAgentStatusShort(t('agentStatusRunning').replace('{type}', typeStr).replace('{lang}', runningShortJobs.join(', ')));
            } else if (isAgentShortActive) {
                 const nextJob = findNextShortJob(shortVideoCadence);
                setAgentStatusShort(nextJob || t('agentStatusIdle').replace('{type}','...').replace('{lang}','...').replace('{time}','...'));
            } else {
                setAgentStatusShort(t('agentStatusDisabled'));
            }
        };

        const checkSchedule = () => {
            const now = new Date();
            const todayStr = now.toISOString().split('T')[0];
            const currentHour = now.getHours();
            const currentMinute = now.getMinutes();

            if (isAgentLongActive) {
                const scheduleHours = schedules['pt'].long.slice(0, longVideoCadence);
                for (const hour of scheduleHours) {
                    const minute = offsets['pt'];
                    const jobKey = `${todayStr}_pt_long_batch_${hour}:${minute}`;
                    if (currentHour === hour && currentMinute === minute && !lastRuns[jobKey]) {
                        setLastRuns(prev => ({ ...prev, [jobKey]: Date.now() }));
                        runAutomatedLongVideoBatch();
                    }
                }
            }
            
            if (isAgentShortActive) {
                for (const lang of ['pt', 'en', 'es'] as const) {
                    const scheduleHours = schedules[lang].short.slice(0, shortVideoCadence);
                    for (const hour of scheduleHours) {
                        const minute = offsets[lang];
                        const jobKey = `${todayStr}_${lang}_short_${hour}:${minute}`;
                        if (currentHour === hour && currentMinute === minute && !lastRuns[jobKey]) {
                            setLastRuns(prev => ({ ...prev, [jobKey]: Date.now() }));
                            runAutomatedShortVideoJob(lang);
                        }
                    }
                }
            }
        };

        updateStatuses();
        const statusInterval = window.setInterval(updateStatuses, 60000);

        let jobIntervalId: number | undefined;
        let startupTimeoutId: number | undefined;

        if (isAgentLongActive || isAgentShortActive) {
            startupTimeoutId = setTimeout(() => {
                checkSchedule();
                jobIntervalId = window.setInterval(checkSchedule, 30000);
            }, 10000);
        }

        return () => {
            clearInterval(statusInterval);
            if (startupTimeoutId) clearTimeout(startupTimeoutId);
            if (jobIntervalId) clearInterval(jobIntervalId);
        };
    }, [isAgentLongActive, isAgentShortActive, longVideoCadence, shortVideoCadence, lastRuns, setLastRuns, runAutomatedLongVideoBatch, runAutomatedShortVideoJob, t, jobsInProgress]);

    const AgentPanel = ({
        type,
        isActive,
        setIsActive,
        cadence,
        setCadence,
        status,
        duration,
        setDuration
    }: {
        type: 'long' | 'short';
        isActive: boolean;
        setIsActive: (val: boolean) => void;
        cadence: number;
        setCadence: (val: number) => void;
        status: string;
        duration?: number;
        setDuration?: (val: number) => void;
    }) => (
         <div className="p-4 bg-gray-900 border border-teal-700 rounded-lg space-y-4 flex flex-col">
            <div className="flex items-start gap-3">
                <BotIcon className="h-6 w-6 text-teal-400 flex-shrink-0 mt-1" />
                <div>
                    <h2 className="text-lg font-bold text-teal-300">{t(type === 'long' ? 'agentTitleLong' : 'agentTitleShort')}</h2>
                    <p className="text-xs text-gray-300">{t(type === 'long' ? 'agentDescriptionLong' : 'agentDescriptionShort')}</p>
                </div>
            </div>
            <div className="flex-grow space-y-3 p-3 bg-gray-800 rounded-lg flex flex-col justify-between">
                <div className="flex items-center gap-3">
                    <label className="font-bold text-gray-200 text-sm">{t('agentStatus')}</label>
                    <label className="flex items-center cursor-pointer">
                        <input type="checkbox" id={`agent-toggle-${type}`} className="sr-only peer" checked={isActive} onChange={() => setIsActive(!isActive)} />
                        <div className="relative w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-teal-600"></div>
                        <span className={`ms-3 text-sm font-medium ${isActive ? 'text-teal-400' : 'text-gray-300'}`}>
                            {isActive ? t('agentStatusActive') : t('agentStatusInactive')}
                        </span>
                    </label>
                </div>

                <div className="flex items-center gap-2">
                    <label className="text-sm font-medium text-gray-200 whitespace-nowrap">{t(type === 'long' ? 'agentCadenceLabel' : 'agentCadenceLabelShort')}:</label>
                    <select
                        value={cadence}
                        onChange={(e) => setCadence(Number(e.target.value))}
                        disabled={!isActive}
                        className="w-full bg-gray-700 text-white p-2 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500 text-sm disabled:opacity-50"
                    >
                        {type === 'long' ? (
                            <>
                                <option value={3}>{t('agentCadence3')}</option>
                                <option value={2}>{t('agentCadence2')}</option>
                                <option value={1}>{t('agentCadence1')}</option>
                            </>
                        ) : (
                             <>
                                <option value={3}>{t('agentCadenceShort3')}</option>
                                <option value={2}>{t('agentCadenceShort2')}</option>
                                <option value={1}>{t('agentCadenceShort1')}</option>
                                <option value={0}>{t('agentCadenceShort0')}</option>
                            </>
                        )}
                    </select>
                </div>

                {type === 'long' && setDuration && (
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-200 whitespace-nowrap">{t('prayerDurationLabel')}:</label>
                        <select
                            value={duration}
                            onChange={(e) => setDuration(Number(e.target.value))}
                            disabled={!isActive}
                            className="w-full bg-gray-700 text-white p-2 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500 text-sm disabled:opacity-50"
                        >
                            <option value={5}>{t('prayerDuration5min')}</option>
                            <option value={10}>{t('prayerDuration10min')}</option>
                            <option value={15}>{t('prayerDuration15min')}</option>
                            <option value={20}>{t('prayerDuration20min')}</option>
                            <option value={30}>30 Minutos (Alta Densidade)</option>
                            <option value={45}>45 Minutos (Alta Densidade)</option>
                            <option value={60}>60 Minutos (Alta Densidade)</option>
                        </select>
                    </div>
                )}

                 <div className="text-xs text-gray-300 italic text-center h-8 flex items-center justify-center">
                    {status.includes(t('agentStatusRunning').split(" ")[0]) ? <SpinnerIcon className="inline-flex w-4 h-4 mr-2" /> : null}
                    {status}
                </div>
            </div>
        </div>
    );

    const IntegrationCard = ({
        platform,
        isConnected,
        icon
    }: {
        platform: 'youtube' | 'tiktok',
        isConnected: boolean,
        icon: React.ReactNode
    }) => (
        <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3">
                <div className={`p-2 rounded-full ${isConnected ? 'bg-green-900/50' : 'bg-gray-800'}`}>
                    {icon}
                </div>
                <div>
                    <h4 className="font-bold text-gray-200 text-sm">{t(platform === 'youtube' ? 'integrationsYouTube' : 'integrationsTikTok')}</h4>
                    <div className="flex items-center gap-1.5">
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                        <span className="text-xs text-gray-300">{isConnected ? t('integrationsConnected') : t('agentStatusInactive')}</span>
                    </div>
                </div>
            </div>
            <button
                onClick={() => isConnected ? handleDisconnect(platform) : handleConnect(platform)}
                disabled={connectingPlatform === platform}
                className={`px-4 py-2 rounded-md text-sm font-bold transition-all ${
                    isConnected 
                        ? 'bg-gray-700 text-gray-300 hover:bg-red-900/30 hover:text-red-400 border border-gray-600' 
                        : 'bg-amber-500 text-gray-900 hover:bg-amber-600'
                } disabled:opacity-50 disabled:cursor-not-allowed w-28 flex justify-center`}
            >
                {connectingPlatform === platform ? (
                    <SpinnerIcon className="h-4 w-4 text-gray-800" />
                ) : (
                    isConnected ? t('integrationsDisconnect') : t('integrationsConnect')
                )}
            </button>
        </div>
    );

    return (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg animate-fade-in space-y-8">
            {/* Agent Controls */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <AgentPanel 
                    type="long"
                    isActive={isAgentLongActive}
                    setIsActive={setIsAgentLongActive}
                    cadence={longVideoCadence}
                    setCadence={setLongVideoCadence}
                    status={agentStatusLong}
                    duration={agentLongDuration}
                    setDuration={setAgentLongDuration}
                />
                <AgentPanel 
                    type="short"
                    isActive={isAgentShortActive}
                    setIsActive={setIsAgentShortActive}
                    cadence={shortVideoCadence}
                    setCadence={setShortVideoCadence}
                    status={agentStatusShort}
                />
            </div>
            <p className="text-center text-xs text-gray-400">{t('agentKeepTabOpen')}</p>

            {/* Integrations Panel */}
            <div className="border-t border-gray-700 pt-6">
                <h3 className="text-xl font-bold text-amber-400 mb-2">{t('integrationsTitle')}</h3>
                <p className="text-sm text-gray-300 mb-4">{t('integrationsDescription')}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <IntegrationCard 
                        platform="youtube" 
                        isConnected={isYoutubeConnected} 
                        icon={<YouTubeIcon className={`h-6 w-6 ${isYoutubeConnected ? 'text-red-500' : 'text-gray-500'}`} />} 
                    />
                    <IntegrationCard 
                        platform="tiktok" 
                        isConnected={isTiktokConnected} 
                        icon={<TikTokIcon className={`h-6 w-6 ${isTiktokConnected ? 'text-pink-500' : 'text-gray-500'}`} />} 
                    />
                </div>
                <p className="text-xs text-gray-400 mt-3 italic text-center">{t('integrationsInfo')}</p>
            </div>
        </div>
    );
};
