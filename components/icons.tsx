
import React from 'react';

export const PrayerIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
    <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.52 4.675a1 1 0 00.95.69h4.905c.969 0 1.371 1.24.588 1.81l-3.97 2.888a1 1 0 00-.364 1.118l1.52 4.675c.3.921-.755 1.688-1.54 1.118l-3.97-2.888a1 1 0 00-1.175 0l-3.97 2.888c-.784.57-1.838-.197-1.539-1.118l1.52-4.675a1 1 0 00-.363-1.118L.98 10.1c-.783-.57-.38-1.81.588-1.81h4.905a1 1 0 00.95-.69L8.951 2.927z" />
  </svg>
);

export const PillIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 3.5a1.5 1.5 0 013 0V4a1 1 0 001 1h.5a1.5 1.5 0 010 3H14a1 1 0 00-1 1v1.5a1.5 1.5 0 01-3 0V10a1 1 0 00-1-1h-.5a1.5 1.5 0 010-3H8a1 1 0 001-1V3.5zM10 6.5a1.5 1.5 0 013 0V7a1 1 0 001 1h.5a1.5 1.5 0 010 3H14a1 1 0 00-1 1v.5a1.5 1.5 0 01-3 0V12a1 1 0 00-1-1h-.5a1.5 1.5 0 010-3H8a1 1 0 001-1v-.5z" />
    </svg>
);

export const ImageIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
  </svg>
);

export const MarketingIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
        <path d="M3 3a1 1 0 000 2v11a1 1 0 100 2h14a1 1 0 100-2V5a1 1 0 000-2H3zm12 1H5v2h10V4zM5 8v2h2V8H5zm4 0v2h2V8H9zm4 0v2h2V8h-2z" />
        <path d="M3 3a1 1 0 00-1 1v.586A1 1 0 003 5V3zM1 15a1 1 0 001 1h.586A1 1 0 001 14.414V15zM17 3a1 1 0 001-1v.586A1 1 0 0017 5V3zM19 15a1 1 0 00-1 1h.586A1 1 0 0019 14.414V15z" />
    </svg>
);

export const SpinnerIcon: React.FC<{ className?: string }> = ({ className = "animate-spin -ml-1 mr-3 h-5 w-5 text-white" }) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

export const LogoIcon: React.FC = () => (
    <svg className="h-10 w-10 text-amber-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
);

export const DownloadIcon: React.FC<{ className?: string }> = ({ className = "h-6 w-6" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
);

export const TrashIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clipRule="evenodd" />
    </svg>
);

export const BotIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 2a8 8 0 100 16 8 8 0 000-16zM6.343 6.343a.5.5 0 01.707 0L10 9.293l2.95-2.95a.5.5 0 11.707.707L10.707 10l2.95 2.95a.5.5 0 01-.707.707L10 10.707l-2.95 2.95a.5.5 0 01-.707-.707L9.293 10 6.343 7.05a.5.5 0 010-.707z" clipRule="evenodd" />
        <path d="M10 2a8 8 0 100 16 8 8 0 000-16zM4.5 10a5.5 5.5 0 1111 0 5.5 5.5 0 01-11 0z" />
    </svg>
);

export const CheckIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
    </svg>
);

export const HistoryIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.414-1.414L11 10.586V6z" clipRule="evenodd" />
    </svg>
);

export const YouTubeIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
    </svg>
);

export const TikTokIcon: React.FC<{ className?: string }> = ({ className = "h-5 w-5" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 1 0 1 13.6 6.84 6.84 0 0 0 6.82-6.85V7.97a10.6 10.6 0 0 0 4.72 1.52v-3.4a7.16 7.16 0 0 1-3.31-.4z"/>
    </svg>
);
