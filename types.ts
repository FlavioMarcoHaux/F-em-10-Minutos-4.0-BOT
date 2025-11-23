

export type AspectRatio = '16:9' | '1:1' | '9:16' | '3:4' | '4:3';

export interface SocialMediaPost {
    title: string;
    description: string;
    hashtags: string[];
}

export interface YouTubeLongPost {
    title: string;
    description: string; 
    hashtags: string[]; // The 3 hashtags for the description field.
    timestamps: string; // The multiline string for video chapters.
    tags: string[]; // The list of tags for the dedicated tags field.
}

export interface MarketingHistoryItem {
    id: string;
    timestamp: number;
    language: string;
    type: 'long' | 'short';
    prompt: string;
    subthemes: string[]; // only for long
    prayer: string;
    socialPost: SocialMediaPost | null;
    longPost: YouTubeLongPost | null;
    audioBlobKey?: string;
    imageBlobKey?: string;
    videoBlobKey?: string;
    isDownloaded?: boolean;
}