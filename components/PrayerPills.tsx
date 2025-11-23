import React from 'react';
import { generateShortPrayer } from '../services/geminiService';
import { PrayerGenerator } from './GuidedPrayer';

export const PrayerPills: React.FC = () => {
    return (
        <PrayerGenerator
            titleKey="pillsTitle"
            descriptionKey="pillsDescription"
            prayerGeneratorFn={generateShortPrayer}
            storageKeyPrefix="prayerPill"
        />
    );
};