
import React, { useState, useMemo, useContext, Dispatch, SetStateAction } from 'react';
import { MarketingHistoryItem } from '../types';
import { LanguageContext, LanguageContextType } from '../context';
import { HistoryItemCard } from './HistoryItemCard';
import { idb } from '../hooks/usePersistentState';

interface MarketingHistoryProps {
    history: MarketingHistoryItem[];
    setHistory: Dispatch<SetStateAction<MarketingHistoryItem[]>>;
    isStandalonePage?: boolean;
}

export const MarketingHistory: React.FC<MarketingHistoryProps> = ({ history, setHistory, isStandalonePage = false }) => {
    const { t } = useContext(LanguageContext) as LanguageContextType;
    const [languageFilter, setLanguageFilter] = useState<'all' | 'pt' | 'en' | 'es'>('all');
    const [searchTerm, setSearchTerm] = useState('');

    const handleDeleteItem = async (id: string) => {
        // Move confirmation dialog here to ensure it's in the correct execution context
        if (window.confirm(t('marketingConfirmDelete'))) {
            const itemToDelete = history.find(item => item.id === id);
            if (!itemToDelete) return;
            
            // Delete associated blobs from IndexedDB
            const blobKeys = [itemToDelete.audioBlobKey, itemToDelete.imageBlobKey, itemToDelete.videoBlobKey];
            const deletePromises = blobKeys.filter(key => !!key).map(key => idb.del(key!));
            
            try {
                await Promise.all(deletePromises);
            } catch (e) {
                console.error("Error deleting blobs from IndexedDB", e);
            }

            // Remove item from history state
            setHistory(prev => prev.filter(item => item.id !== id));
        }
    };

    const filteredHistory = useMemo(() => {
        return history
            .filter(item => {
                if (languageFilter === 'all') return true;
                return item.language === languageFilter;
            })
            .filter(item => {
                if (!searchTerm.trim()) return true;
                const lowerCaseSearch = searchTerm.toLowerCase();
                const title = item.longPost?.title || item.socialPost?.title || '';
                return (
                    title.toLowerCase().includes(lowerCaseSearch) ||
                    item.prompt.toLowerCase().includes(lowerCaseSearch)
                );
            });
    }, [history, languageFilter, searchTerm]);

    const FilterButton: React.FC<{lang: 'all' | 'pt' | 'en' | 'es', label: string}> = ({ lang, label }) => (
        <button
            onClick={() => setLanguageFilter(lang)}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                languageFilter === lang
                    ? 'bg-amber-500 text-gray-900'
                    : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            }`}
        >
            {label}
        </button>
    );

    const containerClasses = isStandalonePage
        ? "bg-gray-800 p-6 rounded-xl shadow-lg animate-fade-in space-y-4"
        : "mt-8 space-y-4";
    
    const titleClasses = isStandalonePage
        ? "text-2xl font-bold text-amber-400"
        : "text-2xl font-bold text-amber-400 border-t border-gray-700 pt-6";

    return (
        <div className={containerClasses}>
            <h3 className={titleClasses}>{t('marketingHistoryTitle')}</h3>
            
            <div className="p-2 bg-gray-900 rounded-lg space-y-3">
                <div className="flex flex-col sm:flex-row gap-4 items-center">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-300">{t('historyFilterLanguage')}</span>
                        <div className="flex gap-1.5 p-1 bg-gray-800 rounded-lg">
                           <FilterButton lang="all" label={t('historyAll')} />
                           <FilterButton lang="pt" label="PT" />
                           <FilterButton lang="en" label="EN" />
                           <FilterButton lang="es" label="ES" />
                        </div>
                    </div>
                    <div className="flex-grow w-full sm:w-auto">
                        <input
                            type="text"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            placeholder={t('historySearchPlaceholder')}
                            className="w-full bg-gray-700 text-white placeholder-gray-400 p-2 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500 transition text-sm"
                        />
                    </div>
                </div>
            </div>

            <div className="space-y-2">
                {filteredHistory.length > 0 ? (
                    filteredHistory.map(item => (
                        <HistoryItemCard key={item.id} item={item} onDelete={handleDeleteItem} setHistory={setHistory}/>
                    ))
                ) : (
                    <div className="text-center py-8 text-gray-400">
                        <p>{t('marketingNoHistory')}</p>
                    </div>
                )}
            </div>
        </div>
    );
};
