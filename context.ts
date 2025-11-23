
import React from 'react';

export interface LanguageContextType {
  language: string;
  setLanguage: (language: string) => void;
  t: (key: string) => string;
}

export const LanguageContext = React.createContext<LanguageContextType | null>(null);
