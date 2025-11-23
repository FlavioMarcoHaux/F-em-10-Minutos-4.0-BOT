
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { AspectRatio, SocialMediaPost, YouTubeLongPost } from "../types";
import { createWavHeader } from "../utils/audio";
import { createOPFSFile } from "../utils/opfsUtils";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export interface MultiSpeakerConfig {
    speakers: { name: string; voice: string }[];
}

export interface SpeechCallbacks {
    onChunk: (data: Uint8Array) => void;
    onProgress: (val: number) => void;
    onComplete: () => void;
    onError: (msg: string) => void;
}

// Helper to sleep between requests to avoid 429
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

interface DialogueBlock {
    speaker: string;
    text: string;
}

/**
 * Parses a script into sequential blocks based on speaker changes.
 * Also handles splitting very long monologues into smaller chunks to fit TTS input limits.
 */
const parseDialogueIntoChunks = (text: string): DialogueBlock[] => {
    const lines = text.split('\n');
    const blocks: DialogueBlock[] = [];
    let currentSpeaker = "Narrator"; // Default
    let currentBuffer = "";

    // Regex to detect "Name: Text" pattern
    const speakerRegex = /^(\*{0,2})([A-Za-z\s]+)(\*{0,2}):\s*(.*)$/;

    const pushBuffer = () => {
        if (currentBuffer.trim()) {
            // Split extremely long buffers to avoid timeouts
            const MAX_CHAR_LIMIT = 1500; // Safe limit for TTS model per request
            if (currentBuffer.length > MAX_CHAR_LIMIT) {
                const sentences = currentBuffer.match(/[^.!?]+[.!?]+["']?|[^.!?]+$/g) || [currentBuffer];
                let tempChunk = "";
                
                for (const sentence of sentences) {
                    if ((tempChunk + sentence).length > MAX_CHAR_LIMIT) {
                        blocks.push({ speaker: currentSpeaker, text: tempChunk.trim() });
                        tempChunk = sentence;
                    } else {
                        tempChunk += sentence;
                    }
                }
                if (tempChunk.trim()) {
                    blocks.push({ speaker: currentSpeaker, text: tempChunk.trim() });
                }
            } else {
                blocks.push({ speaker: currentSpeaker, text: currentBuffer.trim() });
            }
            currentBuffer = "";
        }
    };

    for (const line of lines) {
        const match = line.match(speakerRegex);
        if (match) {
            // New speaker found
            pushBuffer(); // Save previous block
            
            // Clean up speaker name (remove asterisks)
            let rawName = match[2].trim();
            if (rawName.toLowerCase().includes("roberta")) currentSpeaker = "Roberta Erickson";
            else if (rawName.toLowerCase().includes("milton")) currentSpeaker = "Milton Dilts";
            else currentSpeaker = rawName;

            currentBuffer = match[4] + " "; // Start new buffer with the text part
        } else {
            // Continuation of current speaker
            if (line.trim()) {
                currentBuffer += line.trim() + "\n";
            }
        }
    }
    pushBuffer(); // Push last block

    // Fallback: If no speaker structure found, treat entire text as one (or split if huge)
    if (blocks.length === 0 && text.trim().length > 0) {
        currentSpeaker = "Narrator";
        currentBuffer = text;
        pushBuffer();
    }

    return blocks;
};

export const generateGuidedPrayer = async (prompt: string, language: string, duration: number = 10): Promise<string> => {
    // Use gemini-2.5-flash for speed and context window, but orchestrated for long content
    const model = 'gemini-2.5-flash'; 
    
    // BATCH CHAINING STRATEGY:
    // 10 min = ~1500 words. 60 min = ~9000 words.
    // We will generate in chunks of ~10 minutes to ensure density and avoid cutoffs.
    const chunkSizeMinutes = 7; // Safe margin
    const numBatches = Math.ceil(duration / chunkSizeMinutes);
    
    let fullScript = "";
    let previousContext = "";

    console.log(`Generating prayer for ${duration} min in ${numBatches} batches.`);

    for (let i = 1; i <= numBatches; i++) {
        // Define structural focus based on current batch position
        let focus = "";
        if (i === 1) {
            focus = "Introduction (Hypnotic Hook & Validation), Rapport Building, and Initial Induction (Pacing and Leading). Establish the sacred space.";
        } else if (i === numBatches) {
            focus = "Final Integration, Post-Hypnotic Suggestions for future well-being, gratitude, and a Gentle Awakening/Closing Blessing.";
        } else {
            // Middle batches - iterate through subthemes or deepen the main theme
            focus = `Deepening the Trance. Exploring the theme '${prompt}' through metaphors, NLP reframing, and spiritual storytelling. Reinforce the 'Golden Thread' of ${prompt}. Increase emotional intensity. Use embedded commands.`;
        }

        const systemInstruction = `You are a master spiritual scriptwriter, expert in NLP and Ericksonian Hypnosis.
        Language: ${language}.
        
        TASK: Write PARTE ${i} of ${numBatches} for a ${duration}-minute guided prayer/meditation script.
        THEME: ${prompt}.
        GOLDEN THREAD (Anchor): The core theme "${prompt}" must be woven into every segment to maintain focus.
        
        CHARACTERS (STRICT REQUIREMENT):
        The script must be a therapeutic dialogue EXCLUSIVELY between two guides. 
        You MUST use these exact names for the audio engine to work:
        1. **Roberta Erickson** (Voice of Comfort, NLP Guide) -> Maps to 'Aoede' voice.
        2. **Milton Dilts** (Voice of Authority, Hypnotherapist) -> Maps to 'Enceladus' voice.
        
        FORMAT RULES:
        - Output ONLY the dialogue lines.
        - Format: "Speaker Name: Text..."
        - Do NOT use parenthetical stage directions like (softly), (pause), or (music) at the start of lines. Embed the pacing into the punctuation (ellipses...) and italics *emphasis*.
        - Keep the flow continuous.
        - DENSITY: This is for a ${duration} minute audio. Write EXTENSIVELY. Each batch must be long and detailed (approx 1000 words).
        
        CURRENT BATCH FOCUS: ${focus}
        
        ${previousContext ? `CONTEXT FROM PREVIOUS PART: "...${previousContext}"\nContinue smoothly from here, maintaining the narrative arc.` : "Start the session now with a strong hypnotic hook."}
        `;

        const batchPrompt = `Write the next segment of the script (approx 1000 words). Ensure rich sensory details (Visual, Auditory, Kinesthetic) and maintain the "Golden Thread" of ${prompt}.`;

        try {
            console.log(`Requesting batch ${i}/${numBatches}...`);
            const response = await ai.models.generateContent({
                model,
                contents: batchPrompt,
                config: { systemInstruction }
            });
            
            const text = response.text || "";
            fullScript += text + "\n\n";
            
            // Keep the last few sentences as context for the next batch to ensure coherence
            const sentences = text.split('.');
            previousContext = sentences.slice(-3).join('.');
            if (previousContext.length > 500) previousContext = previousContext.substring(previousContext.length - 500);
            
            // Rate limit safety
            if (i < numBatches) await delay(1000);
            
        } catch (error) {
            console.error(`Error generating batch ${i}:`, error);
            if (i === 1) throw error; // If first batch fails, fail all
            break; // Otherwise return what we have
        }
    }

    return fullScript;
};

export const generateShortPrayer = async (prompt: string, language: string): Promise<string> => {
    const model = 'gemini-2.5-flash';
    const systemInstruction = `You are a spiritual companion. Language: ${language}.`;
    const userPrompt = prompt 
        ? `Write a short, powerful prayer (3-5 sentences) about: ${prompt}.`
        : `Write a short, powerful prayer (3-5 sentences) on a random uplifting theme.`;
        
    try {
        const response = await ai.models.generateContent({
            model,
            contents: userPrompt,
            config: { systemInstruction }
        });
        return response.text || "";
    } catch (error) {
        console.error("Error generating short prayer:", error);
        throw error;
    }
};

export const generateSpeech = async (
    text: string, 
    multiSpeakerConfig: MultiSpeakerConfig | undefined, 
    callbacks: SpeechCallbacks, 
    fileHandle?: FileSystemFileHandle
) => {
    try {
        const model = "gemini-2.5-flash-preview-tts";
        
        // 1. PARSE DIALOGUE INTO SEQUENTIAL BLOCKS
        const blocks = parseDialogueIntoChunks(text);
        console.log(`Audio Generation: Split text into ${blocks.length} blocks for sequential processing.`);

        let writable: FileSystemWritableFileStream | undefined;
        let totalSize = 0;
        
        // Prepare OPFS Writer
        if (fileHandle) {
             writable = await fileHandle.createWritable();
             // Placeholder header (44 bytes)
             const header = createWavHeader(0, 1, 24000, 16);
             await writable.write(header);
        }

        // 2. SEQUENTIAL GENERATION LOOP (The "Assembly Line")
        for (let i = 0; i < blocks.length; i++) {
            const block = blocks[i];
            console.log(`Processing Audio Block ${i + 1}/${blocks.length}: ${block.speaker} (${block.text.length} chars)`);

            // Surgical Regex: Removes stage directions ONLY if they appear at the start of the line.
            // Matches: "(Softly) Text" -> "Text" or "[Pause] Text" -> "Text"
            let cleanedText = block.text.replace(/^\s*(?:[\(\[].*?[\)\]])\s*/, "");
            
            // Ensure text is not empty after cleaning
            if (!cleanedText.trim()) continue;

            // Dynamic Voice Selection based on Speaker Name
            let voiceName = 'Kore'; // Default fallback
            const speakerName = block.speaker.toLowerCase();
            
            if (speakerName.includes("roberta")) voiceName = 'Aoede'; // Female, Soft
            else if (speakerName.includes("milton")) voiceName = 'Enceladus'; // Male, Deep
            // Fallback to config if provided
            else if (multiSpeakerConfig) {
                const found = multiSpeakerConfig.speakers.find(s => s.name === block.speaker);
                if (found) voiceName = found.voice;
            }

            const config: any = {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                      prebuiltVoiceConfig: { voiceName: voiceName },
                    },
                },
            };

            // Generate Stream for this specific block
            const responseStream = await ai.models.generateContentStream({
                model,
                contents: cleanedText,
                config
            });

            for await (const chunk of responseStream) {
                const base64Audio = chunk.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
                if (base64Audio) {
                    const binaryString = atob(base64Audio);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let j = 0; j < len; j++) {
                        bytes[j] = binaryString.charCodeAt(j);
                    }

                    // Direct Disk Write (Zero RAM Overhead)
                    if (writable) {
                        await writable.write(bytes);
                    } else {
                        // Fallback for non-fileHandle mode (short audios only)
                        callbacks.onChunk(bytes); 
                    }
                    
                    totalSize += bytes.length;
                }
            }

            // Update Progress
            callbacks.onProgress(Math.round(((i + 1) / blocks.length) * 100));

            // Safety Delay to prevent Rate Limits (429)
            await delay(500); 
        }

        // 3. FINALIZE FILE
        if (writable) {
            // Update WAV header with correct total size
            try {
                const header = createWavHeader(totalSize, 1, 24000, 16);
                await writable.seek(0);
                await writable.write(header);
            } catch (e) {
                console.warn("Could not update WAV header in OPFS", e);
            }
            await writable.close();
        }

        callbacks.onComplete();

    } catch (error: any) {
        callbacks.onError(error.message || "Audio generation failed");
    }
};

export const createMediaPromptFromPrayer = async (prayer: string, language: string): Promise<string> => {
    const model = 'gemini-2.5-flash';
    const response = await ai.models.generateContent({
        model,
        contents: `Based on this prayer, describe a peaceful, spiritual, and artistic image suitable for a background. 
        Prayer: "${prayer.substring(0, 500)}..."
        Output ONLY the English image prompt description. Keep it under 50 words.`,
    });
    return response.text || "Peaceful spiritual landscape";
};

export const generateImageFromPrayer = async (prompt: string, aspectRatio: AspectRatio, modelName: string = 'gemini-2.5-flash-image'): Promise<string> => {
    
    const config: any = {
        aspectRatio: aspectRatio,
    };
    
    if (modelName.includes('imagen')) {
         config.outputMimeType = 'image/jpeg';
         const response = await ai.models.generateImages({
            model: modelName,
            prompt,
            config: {
                ...config,
                numberOfImages: 1,
            }
         });
         return response.generatedImages[0].image.imageBytes;
    } else {
        // Nano banana (gemini-2.5-flash-image)
        const response = await ai.models.generateContent({
            model: modelName,
            contents: { parts: [{ text: prompt }] },
            config: {
                imageConfig: {
                    aspectRatio: aspectRatio as any
                }
            }
        });
        
        for (const part of response.candidates?.[0]?.content?.parts || []) {
            if (part.inlineData) {
                return part.inlineData.data;
            }
        }
        throw new Error("No image data found in response");
    }
};

export const generateVideo = async (prompt: string, aspectRatio: AspectRatio): Promise<string> => {
    let operation = await ai.models.generateVideos({
        model: 'veo-3.1-fast-generate-preview',
        prompt,
        config: {
            numberOfVideos: 1,
            resolution: '720p',
            aspectRatio: aspectRatio as any
        }
    });

    while (!operation.done) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        operation = await ai.operations.getVideosOperation({ operation: operation });
    }

    const uri = operation.response?.generatedVideos?.[0]?.video?.uri;
    if (!uri) throw new Error("Video generation failed to return URI");
    return uri;
};

export const analyzeImage = async (file: File, prompt: string, language: string): Promise<string> => {
    const base64Data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const res = reader.result as string;
            resolve(res.split(',')[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });

    const model = 'gemini-2.5-flash';
    const response = await ai.models.generateContent({
        model,
        contents: {
            parts: [
                { inlineData: { mimeType: file.type, data: base64Data } },
                { text: prompt || `Analyze this image spiritually and symbolically in ${language}.` }
            ]
        }
    });
    return response.text || "";
};

export const createThumbnailPromptFromPost = async (title: string, description: string, prayer: string, language: string): Promise<string> => {
    const model = 'gemini-2.5-flash';
    
    // Determine target language name for prompt instructions
    const langMap: {[key: string]: string} = { 'pt': 'Português', 'en': 'Inglês', 'es': 'Espanhol' };
    const targetLangName = langMap[language] || 'Inglês';

    // SYSTEM INSTRUCTION: VISUAL STRATEGIST FOR CLICKBAIT/HIGH-CTR
    const systemInstruction = `
    You are a world-class YouTube Strategist and Semiotics Expert specialized in High-CTR (Click-Through Rate) Thumbnails.
    
    YOUR GOAL: Generate a single, highly detailed image generation prompt for 'Imagen 4 Ultra' that will result in a VIRAL, CLICKBAIT-STYLE thumbnail.
    
    CRITICAL RULES:
    1. LANGUAGE MATCHING: If the video content is in ${targetLangName}, any text inside the image MUST be in ${targetLangName}. This is non-negotiable.
    2. OUTPUT FORMAT: Return ONLY the raw prompt string in English. Do not add introductions or quotes.
    
    VISUAL FORMULA FOR VIRALITY:
    - **Subject**: Expressive, emotional (e.g., a person in deep prayer with tears of joy, or a divine silhouette against a powerful light).
    - **Text Overlay**: MAXIMUM 3-5 words. Massive, bold, 3D typography. High contrast colors (Yellow/White text on Dark/Blue background).
    - **Effects**: Parallax depth, Glow, God Rays, Particles, High Contrast.
    - **Psychology**: Create a "Curiosity Gap" or a sense of "Immediate Transformation".
    
    The prompt must describe the scene and explicitly instruct the AI to render the text.
    `;

    const userPrompt = `
    CONTEXT:
    Video Title: "${title}"
    Description: "${description.substring(0, 200)}..."
    
    TASK:
    Create the prompt.
    Extract a short, punchy hook from the title (e.g. "GOD SAYS THIS" or "POWERFUL PRAYER") in ${targetLangName} for the text overlay.
    
    Example Structure of your output:
    "A hyper-realistic 8k close-up of [Subject] with [Emotion]. Background is [Atmosphere]. In the foreground, huge 3D glowing text reads: '[HOOK IN ${targetLangName}]'. Cinematic lighting, high contrast, rule of thirds."
    `;

    const response = await ai.models.generateContent({
        model,
        contents: userPrompt,
        config: { systemInstruction }
    });
    return response.text || "Spiritual cinematic background with text overlay";
};

export const generateSocialMediaPost = async (prayer: string, language: string): Promise<SocialMediaPost> => {
     const model = 'gemini-2.5-flash';
     const prompt = `
        You are a social media expert. Create a viral short video post (TikTok/Reels) for this prayer:
        "${prayer.substring(0, 500)}..."
        Language: ${language}.
        
        Return JSON with:
        - title (catchy hook)
        - description (engaging caption)
        - hashtags (array of strings)
     `;
     
     const response = await ai.models.generateContent({
         model,
         contents: prompt,
         config: {
             responseMimeType: "application/json",
             responseSchema: {
                 type: Type.OBJECT,
                 properties: {
                     title: { type: Type.STRING },
                     description: { type: Type.STRING },
                     hashtags: { type: Type.ARRAY, items: { type: Type.STRING } }
                 },
                 required: ["title", "description", "hashtags"]
             }
         }
     });
     
     return JSON.parse(response.text || "{}");
};

export const generateYouTubeLongPost = async (theme: string, subthemes: string[], language: string, durationInMinutes: number = 10): Promise<YouTubeLongPost> => {
    const model = 'gemini-2.5-flash';
    // Scale chapters based on duration (approx 1 chapter every 3-5 mins)
    const numChapters = Math.max(5, Math.ceil(durationInMinutes / 4)); 
    const chaptersStr = `${numChapters}`;

    const prompts: { [key: string]: string } = {
        pt: `
            Você é o especialista em SEO e mídias sociais do canal 'Fé em 10 minutos de Oração' (YouTube: https://www.youtube.com/@fe10minutos).
            Sua tarefa é gerar Título, Descrição, Capítulos e Tags otimizados para um vídeo longo de oração (${durationInMinutes} min) sobre "${theme}".
            Subtemas: ${subthemes.join(', ')}.
            
            REGRAS (TÍTULO):
            - Deve ser chamativo, usar emoção/urgência e conter "${theme}".
            - Modelo: "ORAÇÃO PODEROSA DE ${durationInMinutes} MINUTOS para [TEMA]" ou "A ORAÇÃO MAIS PODEROSA para [TEMA]".
            - Deve terminar com: "| Fé em 10 minutos".

            REGRAS (DESCRIÇÃO):
            1. Comece repetindo exatamente o Título.
            2. Escreva uma descrição rica (300-500 palavras) usando técnicas de PNL e Copywriting para prender a atenção, focada em "oração poderosa", "conversa com Deus" e "${theme}".
            3. OBRIGATÓRIO - Inclua EXATAMENTE estes links no final da descrição (não altere nada nos links):

            🌌 PARTICIPE DESTA JORNADA:
            ► SÉRIE: Portais da Consciência (Playlist): https://www.youtube.com/watch?v=Q6x_C3uaKsQ&list=PLmeEfeSNeLbIyeBMB8HLrHwybI__suhgq
            ► SÉRIE: ARQUITETURA DA ALMA (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbIIm3MzGHSRFYfIONlBDofI
            ► Oração da Manhã (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbKppEyZUaBoXw4BVxZTq-I2
            ► Oração da Noite (Playlist): https://www.youtube.com/playlist?list=PLmeEfeSNeLbLFUayT8Sfb9IQzr0ddkrHC
            🔗 INSCREVA-SE NO CANAL: https://www.youtube.com/@fe10minutos

            REGRAS (CAPÍTULOS):
            - Gere uma lista de ${chaptersStr} títulos inspiradores baseados nos subtemas.
            - **APENAS OS TÍTULOS. NÃO COLOQUE MINUTAGEM (ex: 00:00).**

            REGRAS (TAGS/HASHTAGS):
            - 3 Hashtags principais na descrição: #Oração #Fé #[TEMA_Sem_Espaço]
            - Tags (campo tags): Lista com pelo menos 20 tags, incluindo: Fé em 10 minutos, Oração de 10 minutos, Oração Poderosa, ${theme}, Oração do Dia, Oração Guiada, Falar com Deus, Oração da Manhã, Oração da Noite, Palavra de Deus, Espiritualidade, Bênção, Milagre, Cura, Libertação.
            
            Retorne APENAS um JSON com: title, description, hashtags (array), timestamps (string multilinhas - SÓ TÍTULOS), tags (array).
        `,
        en: `
            You are the SEO and social media expert for the 'Faith in 10 Minutes' channel (YouTube: https://www.youtube.com/@Faithin10Minutes).
            Your task is to generate an optimized Title, Description, Timestamps, and Tags for a new long-form video (${durationInMinutes} minutes) on "${theme}".
            Subtopics: ${subthemes.join(', ')}.

            RULES (TITLE):
            - Must be catchy, use emotion/urgency, and contain "${theme}".
            - Follow model: "POWERFUL ${durationInMinutes}-MINUTE PRAYER for [TOPIC]" or "THE MOST POWERFUL PRAYER for [TOPIC]".
            - Must end with: "| Faith in 10 Minutes".

            RULES (DESCRIPTION):
            1. Start by repeating the exact Title.
            2. Write a rich description (300-500 words) using keywords: "powerful prayer", "guided prayer", "relationship with God", "message of faith", and "${theme}".
            3. Include CTA links EXACTLY as follows at the end:

            🕊️ WATCH NEXT:
            ► Architecture of the Soul (Playlist) https://www.youtube.com/playlist?list=PLTQIQ5QpCYPo11ap1JUSiItZtoiV_4lEH
            ► Morning Prayers (Playlist): https://www.youtube.com/playlist?list=PLTQIQ5QpCYPqym_6TF19PB71SpLpAGuZr
            ► Evening Prayers (Playlist): https://www.youtube.com/playlist?list=PLTQIQ5QpCYPq91fvXaDSideb8wrnG-YtR
            🔗 SUBSCRIBE TO THE CHANNEL: https://www.youtube.com/@Faithin10Minutes

            RULES (TIMESTAMPS):
            - Generate a list of ${chaptersStr} inspiring chapter titles based on subtopics.
            - **TITLES ONLY. DO NOT INCLUDE TIMESTAMPS (e.g., 00:00).**

            RULES (TAGS/HASHTAGS):
            - 3 hashtags in description: #Prayer #Faith #[TOPIC_No_Space]
            - Tags field: List at least 20 tags, including: Faith in 10 Minutes, 10 Minute Prayer, Powerful Prayer, ${theme}, Daily Prayer, Guided Prayer, Relationship with God, Morning Prayer, Evening Prayer, Prayer for Sleep, Prayer for Anxiety, Prayer for Healing, Word of God, Spirituality, Blessing, Miracle.
            
            Return ONLY JSON with: title, description, hashtags (array), timestamps (multiline string - TITLES ONLY), tags (array).
        `,
        es: `
            Genera metadatos de YouTube OPTIMIZADOS PARA SEO y de ALTA CONVERSIÓN para un video de oración de ${durationInMinutes} min sobre "${theme}".
            Subtemas: ${subthemes.join(', ')}.
            
            **REGLAS OBLIGATORIAS:**
            1. Título Viral y Emocional: Usa gatillos mentales. Ej: "LA ORACIÓN MÁS PODEROSA DE [TEMA] PARA CAMBIAR TU VIDA | Fe en 10 Minutos".
            2. Descripción Rica (Copywriting): Escribe una descripción de 300-500 palabras. Usa técnicas de PNL e Hipnosis en el texto para captar la atención. Incluye Llamada a la Acción (Suscríbete, Comenta).
            3. Enlaces Obligatorios (Al final de la descripción):
            🔗 SUSCRÍBETE AL CANAL: https://www.youtube.com/@Faithin10Minutes
            4. Capítulos: Genera una lista de ${chaptersStr} títulos de capítulos inspiradores. **SOLO LOS TÍTULOS. NO INCLUYAS TIEMPOS (ej: 00:00).**
            5. Hashtags: 3 hashtags de alto volumen en la descripción.
            6. Tags: Lista de 20 etiquetas SEO (Fe en 10 minutos, Oración Poderosa, etc.).
            
            Retorna SOLO un JSON con: title, description, hashtags (array), timestamps (string multilínea - SOLO TÍTULOS), tags (array).
        `
    };
    
    const prompt = prompts[language] || prompts['en'];

    const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                 type: Type.OBJECT,
                 properties: {
                     title: { type: Type.STRING },
                     description: { type: Type.STRING },
                     hashtags: { type: Type.ARRAY, items: { type: Type.STRING } },
                     timestamps: { type: Type.STRING },
                     tags: { type: Type.ARRAY, items: { type: Type.STRING } }
                 },
                 required: ["title", "description", "hashtags", "timestamps", "tags"]
             }
        }
    });
    return JSON.parse(response.text || "{}");
};

export const getTrendingTopic = async (language: string, type: 'long' | 'short'): Promise<{ theme: string; subthemes: string[] }> => {
    const model = 'gemini-2.5-flash';
    const prompt = `
        Identify a trending or timeless spiritual topic suitable for a ${type === 'long' ? 'YouTube video' : 'TikTok/Reels'} today. 
        Language: ${language}.
        Target Audience: People seeking peace, faith, or strength.
        
        Return JSON:
        - theme (string)
        - subthemes (array of 3 strings)
    `;
    
    const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                 type: Type.OBJECT,
                 properties: {
                     theme: { type: Type.STRING },
                     subthemes: { type: Type.ARRAY, items: { type: Type.STRING } }
                 },
                 required: ["theme", "subthemes"]
             }
        }
    });
    return JSON.parse(response.text || "{}");
};
