
import React, { useState, useContext } from 'react';
import { GuidedPrayer } from './components/GuidedPrayer';
import { ImageAnalyzer } from './components/ImageAnalyzer';
import { PrayerPills } from './components/PrayerPills';
import { MarketingExpert } from './components/MarketingExpert';
import { BotAgent } from './components/BotAgent';
import { MarketingHistory } from './components/MarketingHistory';
import { PrayerIcon, ImageIcon, LogoIcon, PillIcon, MarketingIcon, BotIcon, HistoryIcon } from './components/icons';
import { LanguageContext, LanguageContextType } from './context';
import { getTranslator, supportedLanguages } from './i18n';
import { usePersistentState } from './hooks/usePersistentState';
import { MarketingHistoryItem } from './types';


type View = 'prayer' | 'pills' | 'image' | 'marketing' | 'bot' | 'history';

const ViewWrapper = ({ view, activeView, children }: { view: View; activeView: View; children: React.ReactNode }) => (
    <div style={{ display: view === activeView ? 'block' : 'none' }}>
        {children}
    </div>
);

const AppContent: React.FC = () => {
  const [activeView, setActiveView] = useState<View>('prayer');
  const { t } = useContext(LanguageContext) as LanguageContextType;
  const [history, setHistory] = usePersistentState<MarketingHistoryItem[]>('marketing_history', []);


  const NavButton = ({ view, labelKey, icon }: { view: View; labelKey: string; icon: React.ReactNode }) => (
    <button
      onClick={() => setActiveView(view)}
      className={`flex-1 flex flex-col sm:flex-row items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-amber-400 rounded-lg ${
        activeView === view
          ? 'bg-amber-500 text-gray-900 shadow-lg'
          : 'bg-gray-700 text-gray-200 hover:bg-gray-600 hover:text-white'
      }`}
    >
      {icon}
      <span>{t(labelKey)}</span>
    </button>
  );

  const LanguageSwitcher: React.FC = () => {
    const { language, setLanguage } = useContext(LanguageContext) as LanguageContextType;
    return (
      <div className="absolute top-4 right-4 z-10">
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="bg-gray-700 text-white p-2 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500 text-sm"
          aria-label={t('selectLanguage')}
        >
          {supportedLanguages.map(({ code, name }) => (
            <option key={code} value={code}>
              {name}
            </option>
          ))}
        </select>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl relative">
      <LanguageSwitcher />
      <header className="text-center mb-6 pt-12 sm:pt-0">
        <div className="flex items-center justify-center gap-3 mb-2">
          <LogoIcon />
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-amber-300 to-amber-500">
            {t('appTitle')}
          </h1>
        </div>
        <p className="text-gray-300">{t('appSubtitle')}</p>
      </header>

      <nav className="flex flex-wrap gap-2 sm:gap-4 p-2 bg-gray-800 rounded-xl shadow-md mb-6">
        <NavButton view="prayer" labelKey="navPrayer" icon={<PrayerIcon />} />
        <NavButton view="pills" labelKey="navPills" icon={<PillIcon />} />
        <NavButton view="image" labelKey="navImage" icon={<ImageIcon />} />
        <NavButton view="marketing" labelKey="navMarketing" icon={<MarketingIcon />} />
        <NavButton view="bot" labelKey="navBot" icon={<BotIcon />} />
        <NavButton view="history" labelKey="navHistory" icon={<HistoryIcon />} />
      </nav>

      <main>
        <ViewWrapper view="prayer" activeView={activeView} children={<GuidedPrayer />} />
        <ViewWrapper view="pills" activeView={activeView} children={<PrayerPills />} />
        <ViewWrapper view="image" activeView={activeView} children={<ImageAnalyzer />} />
        <ViewWrapper view="marketing" activeView={activeView} children={<MarketingExpert history={history} setHistory={setHistory} />} />
        {/* Fix: Pass children as an explicit prop to satisfy the component's type definition and maintain consistency with other usages. */}
        <ViewWrapper view="bot" activeView={activeView} children={<>
          <BotAgent history={history} setHistory={setHistory} />
          <MarketingHistory history={history} setHistory={setHistory} isStandalonePage={false} />
        </>} />
        <ViewWrapper view="history" activeView={activeView} children={<MarketingHistory history={history} setHistory={setHistory} isStandalonePage={true} />} />
      </main>
      
      <footer className="text-center mt-8 text-gray-400 text-xs">
          <p>{t('footerCommunity')}</p>
          <div className="flex justify-center gap-4 mt-2">
              <a href="https://www.youtube.com/@fe10minutos" target="_blank" rel="noopener noreferrer" className="hover:text-amber-400 transition-colors">Fé em 10 Minutos</a>
              <a href="https://www.youtube.com/@Faithin10Minutes" target="_blank" rel="noopener noreferrer" className="hover:text-amber-400 transition-colors">Faith in 10 Minutes</a>
          </div>
      </footer>
    </div>
  );
};

const App: React.FC = () => {
  const [language, setLanguage] = useState(() => {
    return localStorage.getItem('appLanguage') || 'pt';
  });
  
  const t = getTranslator(language);

  const handleSetLanguage = (lang: string) => {
    setLanguage(lang);
    localStorage.setItem('appLanguage', lang);
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900 font-sans">
        <AppContent />
      </div>
    </LanguageContext.Provider>
  );
};

export default App;
