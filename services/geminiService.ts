
import { GoogleGenAI, GenerateContentResponse, Part } from "@google/genai";
import { createOPFSFile, writeChunkToStream, getOPFSFileAsBlob } from '../utils/opfsUtils';
import { YouTubeLongPost, SocialMediaPost, AspectRatio } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Helper to clean stage directions from the start of lines for TTS
const cleanTextForSpeech = (text: string): string => {
    // Removes (Softly), [Pause], etc., only if they appear at the start of a line or sentence
    // Preserves internal emphasis like *stars* or (biblical references) inside the sentence.
    return text.replace(/^\s*[\(\[][^)\]]*[\)\]]\s*/gm, "");
};

export interface MultiSpeakerConfig {
    speakers: { name: string; voice: string }[];
}

// --- CORE GENERATION FUNCTIONS ---

export const generateGuidedPrayer = async (prompt: string, language: string, duration: number = 10): Promise<string> => {
    const model = 'gemini-2.5-flash'; // Using Flash for high-volume text generation (recursion)
    
    // Language Map
    const langMap: {[key: string]: string} = { 'pt': 'Português', 'en': 'Inglês', 'es': 'Espanhol' };
    const targetLang = langMap[language] || 'Inglês';

    // Calculate iterations based on duration to ensure density
    // 10 min = 2 calls (approx 2500 words) -> High Density
    // 60 min = 8 calls (approx 10000 words)
    const numIterations = Math.ceil(duration / 7); 
    let fullPrayer = "";
    let lastContext = "";

    console.log(`Starting Recursive Generation: ${duration} min = ${numIterations} iterations.`);

    for (let i = 0; i < numIterations; i++) {
        const isFirst = i === 0;
        const isLast = i === numIterations - 1;
        
        const systemInstruction = `
        You are a Master of Guided Prayer and Erickson Hypnosis.
        Your goal is to write a DEEPLY THERAPEUTIC dialogue script.
        
        CRITICAL RULES:
        1. CHARACTERS: The dialogue MUST be exclusively between "Roberta Erickson" (Voice: Aoede, Soft, NLP Guide) and "Milton Dilts" (Voice: Enceladus, Deep, Hypnotic Voice).
        2. FORMAT: Always start lines with "Roberta Erickson:" or "Milton Dilts:". Do NOT use other names.
        3. LANGUAGE: Write strictly in ${targetLang}.
        4. NO META-DATA: Do NOT write introductions like "Here is the script", summaries, or stage directions in parentheses at the start of lines. Just the dialogue.
        5. DENSITY: Write extensive, rich, poetic text. Use sensory descriptions (VAK), loops, and embedded commands.
        6. GOLDEN THREAD: The central theme "${prompt || 'Divine Connection'}" must be woven into every paragraph to maintain focus.
        
        STRUCTURAL GOAL FOR THIS BLOCK (Part ${i + 1} of ${numIterations}):
        ${isFirst ? "- Start with a 'Hypnotic Hook': A provocative question or deep validation of the user's pain to grab attention immediately (First 30s). Then move to induction." : ""}
        ${!isFirst && !isLast ? "- Deepening: Biblical metaphors (David/Solomon/Jesus), PNL ressignification, sensory immersion. Expand on the theme." : ""}
        ${isLast ? "- Anchor the feeling, gratitude, and slowly return. End with a blessing." : ""}
        
        ${!isFirst ? `CONTEXT FROM PREVIOUS BLOCK: "...${lastContext.slice(-300)}"` : ""}
        `;

        const userPrompt = `
        Write Part ${i + 1}/${numIterations} of the prayer about "${prompt}".
        Duration target for this block: ~8 minutes of spoken text (approx 1200 words).
        Keep the flow continuous. Start directly with a character name.
        `;

        try {
            const result = await ai.models.generateContent({
                model,
                contents: userPrompt,
                config: { systemInstruction, temperature: 0.7 } // Creative but coherent
            });
            
            const text = result.text || "";
            fullPrayer += text + "\n\n";
            lastContext = text;
        } catch (e) {
            console.error(`Error in block ${i}:`, e);
            // If one block fails, we return what we have so far rather than crashing
            break; 
        }
    }

    return fullPrayer;
};

export const generateShortPrayer = async (prompt: string, language: string): Promise<string> => {
    // Short prayer (pills) doesn't need recursion
    return generateGuidedPrayer(prompt, language, 5); 
};

// --- SPEECH GENERATION (BLADE RUNNER ARCHITECTURE) ---

const parseDialogueIntoChunks = (text: string): { speaker: string; text: string }[] => {
    const lines = text.split('\n');
    const chunks: { speaker: string; text: string }[] = [];
    let currentSpeaker = 'Narrator'; // Default
    let currentBuffer = '';

    // Regex to detect "Name:" pattern
    const speakerRegex = /^([A-Za-zÀ-ÖØ-öø-ÿ ]+):/i;

    for (const line of lines) {
        const match = line.match(speakerRegex);
        if (match) {
            // If we have a buffer for the previous speaker, push it
            if (currentBuffer.trim()) {
                chunks.push({ speaker: currentSpeaker, text: currentBuffer.trim() });
            }
            // Start new speaker
            currentSpeaker = match[1].trim();
            currentBuffer = line.replace(speakerRegex, '').trim();
        } else {
            // Append to current speaker
            if (line.trim()) {
                currentBuffer += ' ' + line.trim();
            }
        }
    }
    // Push final buffer
    if (currentBuffer.trim()) {
        chunks.push({ speaker: currentSpeaker, text: currentBuffer.trim() });
    }

    // Further split very long chunks to avoid TTS timeouts
    const finalChunks: { speaker: string; text: string }[] = [];
    const MAX_CHARS = 1500; 

    for (const chunk of chunks) {
        if (chunk.text.length > MAX_CHARS) {
            // Split by sentences roughly
            const sentences = chunk.text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [chunk.text];
            let temp = '';
            for (const sentence of sentences) {
                if ((temp + sentence).length > MAX_CHARS) {
                    finalChunks.push({ speaker: chunk.speaker, text: temp });
                    temp = sentence;
                } else {
                    temp += temp ? ' ' + sentence : sentence;
                }
            }
            if (temp) finalChunks.push({ speaker: chunk.speaker, text: temp });
        } else {
            finalChunks.push(chunk);
        }
    }

    return finalChunks;
};

export const generateSpeech = async (
    text: string, 
    multiSpeakerConfig?: MultiSpeakerConfig,
    callbacks?: {
        onChunk?: (data: Uint8Array) => void,
        onProgress?: (progress: number) => void,
        onComplete?: () => void,
        onError?: (msg: string) => void
    },
    opfsFileHandle?: FileSystemFileHandle
): Promise<void> => {
    const model = 'gemini-2.5-flash-preview-tts'; // Correct TTS model
    const blocks = parseDialogueIntoChunks(text);
    const totalBlocks = blocks.length;
    let processedBlocks = 0;

    // Open writable stream if OPFS is used
    let writable: FileSystemWritableFileStream | null = null;
    if (opfsFileHandle) {
        writable = await opfsFileHandle.createWritable({ keepExistingData: false });
    }

    for (const block of blocks) {
        try {
            // Determine voice
            let voiceName = 'Aoede'; // Default female
            if (multiSpeakerConfig) {
                const speakerMap = multiSpeakerConfig.speakers.find(s => 
                    block.speaker.toLowerCase().includes(s.name.toLowerCase().split(' ')[0]) // Match first name
                );
                if (speakerMap) voiceName = speakerMap.voice;
            }

            // Surgical Clean: Remove stage directions from start of speech only
            const textToSpeak = cleanTextForSpeech(block.text);
            if (!textToSpeak.trim()) continue;

            const response = await ai.models.generateContent({
                model,
                contents: { parts: [{ text: textToSpeak }] },
                config: {
                    responseModalities: ['AUDIO'],
                    speechConfig: {
                        voiceConfig: { prebuiltVoiceConfig: { voiceName } }
                    }
                }
            });

            const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            
            if (audioData) {
                // Decode Base64 to Uint8Array
                const binaryString = atob(audioData);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }

                if (writable) {
                    await writeChunkToStream(writable, bytes);
                } else if (callbacks?.onChunk) {
                    callbacks.onChunk(bytes);
                }
            }

            processedBlocks++;
            if (callbacks?.onProgress) {
                callbacks.onProgress(Math.round((processedBlocks / totalBlocks) * 100));
            }

            // Small delay to be gentle on rate limits
            await new Promise(r => setTimeout(r, 100));

        } catch (e: any) {
            console.error("TTS Generation Error on block:", block, e);
            if (callbacks?.onError) callbacks.onError(`Error generating audio for block ${processedBlocks + 1}`);
            // Continue to next block to salvage what we can
        }
    }

    if (writable) {
        await writable.close();
    }

    if (callbacks?.onComplete) callbacks.onComplete();
};

// --- VISUAL GENERATION ---

export const createThumbnailPromptFromPost = async (title: string, description: string, prayer: string, language: string): Promise<string> => {
    const model = 'gemini-2.5-flash';
    const langMap: {[key: string]: string} = { 'pt': 'Português', 'en': 'Inglês', 'es': 'Espanhol' };
    const targetLangName = langMap[language] || 'Inglês';

    const systemInstruction = `
    You are a world-class YouTube Strategist and Semiotics Expert, specialized in 'SEXY CANVAS' psychology to create High-CTR Thumbnails.
    
    YOUR GOAL: Generate a prompt for 'Imagen 4 Ultra' to create a VIRAL, CLICKBAIT-STYLE thumbnail based **STRICTLY** on the Marketing TITLE.
    
    CRITICAL RULES:
    1. SOURCE OF TRUTH: Analyze **ONLY the TITLE** to determine the hook. Do NOT look at the description or prayer text for the text overlay content.
    2. LANGUAGE MATCHING: Text inside the image MUST be in ${targetLangName}.
    3. OUTPUT FORMAT: Return ONLY the raw prompt string in English.
    4. TEXT STRUCTURE: The text overlay MUST consist of TWO SHORT PHRASES (Headline + Subheadline). The Subheadline MUST have at least 3 words. Use synonyms from the title to avoid exact repetition.
    
    SEXY CANVAS METHODOLOGY (Analyze the TITLE to choose the trigger):
    - **Sloth (Laziness)**: If title promises fast results ("1 Minute"). Text Ex: "DURMA AGORA / PAZ INSTANTÂNEA AQUI".
    - **Greed (Gain)**: If title promises blessings/money. Text Ex: "RECEBA TUDO / MILAGRE FINANCEIRO HOJE".
    - **Wrath (Justice)**: If title mentions enemies. Text Ex: "ELES CAIRÃO / FOGO CONTRA O MAL".
    - **Pride (Chosen)**: If title says "God chose you". Text Ex: "VOCÊ FOI / ESCOLHIDO POR DEUS".
    - **Lust (Intimacy)**: If title talks about Love. Text Ex: "AMOR REAL / ELE TE OUVE AGORA".
    
    VISUAL FORMULA:
    - **Subject**: Highly expressive human face (close up) showing emotion relevant to the hook OR Divine/Mystical silhouette with glowing aura.
    - **Text Overlay**: Massive 3D font, High Contrast (Yellow/White on Dark).
    - **Style**: Hyper-realistic, 8k, cinematic lighting, YouTube Clickbait style (MrBeast style high contrast).
    `;

    const userPrompt = `
    MARKETING TITLE: "${title}"
    (Analyze ONLY this Title for the visual hook and text).
    
    CONTEXT (Mood only - DO NOT use for text):
    Prayer Theme: "${prayer.substring(0, 100)}..."
    
    TASK:
    1. Extract the 'Sexy Canvas' trigger from the TITLE.
    2. Define the text overlay: TWO PHRASES in ${targetLangName}. Subtitle must be 3+ words. Use synonyms.
    3. Generate the full image prompt describing the visual and the specific text to render.
    `;

    const response = await ai.models.generateContent({
        model,
        contents: userPrompt,
        config: { systemInstruction }
    });
    return response.text || "Spiritual cinematic background with text overlay";
};

export const createMediaPromptFromPrayer = async (prayer: string, language: string): Promise<string> => {
    // This is for the 'Video Background' or 'Art', not the Thumbnail. 
    // It should be more artistic and less 'clickbaity'.
    const model = 'gemini-2.5-flash';
    const prompt = `
    Create a prompt for an AI image generator to create a cinematic, spiritual background image 
    that matches the themes of this prayer: "${prayer.substring(0, 500)}...".
    Style: Ethereal, hyper-realistic, 8k, cinematic lighting, peaceful, divine atmosphere.
    No text in the image.
    Return ONLY the prompt in English.
    `;
    const response = await ai.models.generateContent({ model, contents: prompt });
    return response.text || "Ethereal spiritual background, cinematic lighting, 8k";
};

export const generateImageFromPrayer = async (prompt: string, aspectRatio: AspectRatio, model: string = 'imagen-3.0-generate-001'): Promise<string> => {
    const response = await ai.models.generateImages({
        model,
        prompt,
        config: {
            numberOfImages: 1,
            aspectRatio: aspectRatio,
            outputMimeType: 'image/png',
        },
    });
    return response.generatedImages[0].image.imageBytes;
};

export const generateVideo = async (prompt: string, aspectRatio: AspectRatio): Promise<string> => {
    // Video generation is expensive/slow, ensure we use the correct model
    const model = 'veo-3.1-fast-generate-preview'; 
    let operation = await ai.models.generateVideos({
        model,
        prompt,
        config: {
            numberOfVideos: 1,
            resolution: '720p',
            aspectRatio: aspectRatio
        }
    });
    
    // Poll for completion
    while (!operation.done) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        operation = await ai.operations.getVideosOperation({ operation: operation });
    }
    
    return operation.response?.generatedVideos?.[0]?.video?.uri || "";
};

// --- MARKETING ASSETS GENERATION ---

export const generateSocialMediaPost = async (prayer: string, language: string): Promise<SocialMediaPost> => {
    const model = 'gemini-2.5-flash';
    const prompt = `
    You are a Social Media Manager for a spiritual channel.
    Create a viral Instagram/TikTok caption for this prayer: "${prayer.substring(0, 500)}..."
    Language: ${language}
    
    Output format JSON:
    {
        "title": "Catchy Hook (Max 50 chars)",
        "description": "Engaging caption with emojis (Max 300 chars)",
        "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }
    `;
    
    const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: { responseMimeType: "application/json" }
    });
    
    return JSON.parse(response.text || "{}");
};

export const generateYouTubeLongPost = async (theme: string, subthemes: string[], language: string, duration: number): Promise<YouTubeLongPost> => {
    const model = 'gemini-2.5-flash';
    const isPT = language === 'pt';
    
    // Define Static Blocks based on Language (Strict Identity)
    const linksBlock = isPT ? `
🌌 PARTICIPE DESTA JORNADA:

► SÉRIE: Portais da Consciência (Playlist): [https://www.youtube.com/watch?v=Q6x_C3uaKsQ&list=PLmeEfeSNeLbIyeBMB8HLrHwybI__suhgq]

► SÉRIE: ARQUITETURA DA ALMA (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbIIm3MzGHSRFYfIONlBDofI

► Oração da Manhã (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbKppEyZUaBoXw4BVxZTq-I2

► Oração da Noite (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbLFUayT8Sfb9IQzr0ddkrHC

🔗 INSCREVA-SE NO CANAL: https://www.youtube.com/@fe10minutos
    ` : `
🕊️ WATCH NEXT:

► Architecture of the Soul (Playlist) https://www.youtube.com/playlist?list=PLTQIQ5QpCYPo11ap1JUSiItZtoiV_4lEH

► Morning Prayers (Playlist): https://www.youtube.com/playlist?list=PLTQIQ5QpCYPqym_6TF19PB71SpLpAGuZr

► Evening Prayers (Playlist): https://www.youtube.com/playlist?list=PLTQIQ5QpCYPq91fvXaDSideb8wrnG-YtR

🔗 SUBSCRIBE TO THE CHANNEL: https://www.youtube.com/@Faithin10Minutes
    `;

    const systemInstruction = `
    You are the SEO Expert for the channel '${isPT ? 'Fé em 10 Minutos' : 'Faith in 10 Minutes'}'.
    Task: Create metadata for a ${duration}-minute guided prayer video about "${theme}".
    
    CRITICAL OUTPUT RULES:
    1. **Title**: Must be CLICKBAIT/High-Urgency. Use CAPS and Emojis. Model: "POWERFUL ${duration} MIN PRAYER for [TOPIC] | ${isPT ? 'Fé em 10 Minutos' : 'Faith in 10 Minutes'}".
    2. **Description**: 
       - Paragraph 1: AIDA Copywriting hook (3 sentences). Start by repeating the exact Title.
       - Paragraph 2: Describe the prayer using keywords: "powerful prayer", "guided prayer", "relationship with God".
       - **MANDATORY**: Insert the LINKS BLOCK exactly as provided below (Do not translate URLs or change format).
       - End with 3 strong hashtags: #Prayer #Faith #[TOPIC_No_Space]
    3. **Tags**: Generate 20 high-volume tags mixed with long-tail keywords (e.g., Faith in 10 Minutes, ${duration} Minute Prayer, Powerful Prayer, [TOPIC], Daily Prayer).
    4. **Timestamps**: Generate a list of chapters based on the subthemes. **DO NOT INCLUDE TIME CODES (00:00)**. Just the list of topics (e.g., "Introduction", "Prayer for [Subtheme 1]").
    
    MANDATORY LINKS BLOCK TO INSERT IN DESCRIPTION:
    ${linksBlock}
    `;

    const prompt = `
    Generate JSON for this video:
    Theme: ${theme}
    Subthemes: ${subthemes.join(', ')}
    
    Output Schema:
    {
        "title": "string",
        "description": "string (including the links block)",
        "hashtags": ["#string", "#string", "#string"],
        "timestamps": "string (multiline list of topics, NO TIME CODES)",
        "tags": ["string", "string", ...]
    }
    `;

    const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: { 
            responseMimeType: "application/json",
            systemInstruction
        }
    });

    return JSON.parse(response.text || "{}");
};

// --- ANALYSIS FUNCTIONS ---

export const analyzeImage = async (imageFile: File, prompt: string, language: string): Promise<string> => {
    const model = "gemini-2.5-flash"; 
    
    const base64Image = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(imageFile);
    });
    const data = base64Image.split(',')[1];

    const userPrompt = prompt || (language === 'pt' ? "Analise esta imagem espiritualmente." : "Analyze this image spiritually.");

    const response = await ai.models.generateContent({
        model,
        contents: {
            parts: [
                { inlineData: { mimeType: imageFile.type, data } },
                { text: userPrompt }
            ]
        }
    });

    return response.text || "";
};

export const getTrendingTopic = async (language: string, type: 'long' | 'short'): Promise<{theme: string, subthemes: string[]}> => {
    // Simulated Trending Topics for the Agent
    const themes = language === 'pt' 
        ? ['Cura da Ansiedade', 'Prosperidade Financeira', 'Dormir em Paz', 'Proteção da Família', 'Gratidão Matinal']
        : ['Healing Anxiety', 'Financial Prosperity', 'Sleep in Peace', 'Family Protection', 'Morning Gratitude'];
    
    const randomTheme = themes[Math.floor(Math.random() * themes.length)];
    
    return {
        theme: randomTheme,
        subthemes: ['Introduction', 'Deep Dive', 'Closing']
    };
};
