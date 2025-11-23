import { useState, useEffect, Dispatch, SetStateAction } from 'react';

export function usePersistentState<T>(key: string, defaultValue: T): [T, Dispatch<SetStateAction<T>>] {
    const [state, setState] = useState<T>(() => {
        try {
            const storedValue = localStorage.getItem(key);
            if (storedValue) {
                return JSON.parse(storedValue);
            }
        } catch (error) {
            console.error(`Error reading localStorage key “${key}”:`, error);
        }
        return defaultValue;
    });

    useEffect(() => {
        try {
            if (state === defaultValue && !localStorage.getItem(key)) {
                return;
            }
            localStorage.setItem(key, JSON.stringify(state));
        } catch (error) {
            console.error(`Error writing to localStorage key “${key}”:`, error);
        }
    }, [key, state, defaultValue]);

    return [state, setState];
}

// A simple key-val store for IndexedDB to handle larger data like blobs
export const idb = {
  get: <T>(key: string): Promise<T | undefined> => {
    return new Promise((resolve) => {
      const request = indexedDB.open('persistent-blob-store', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('blobs');
      request.onsuccess = () => {
        const db = request.result;
        try {
            const tx = db.transaction('blobs', 'readonly');
            const store = tx.objectStore('blobs');
            const getRequest = store.get(key);
            getRequest.onsuccess = () => resolve(getRequest.result);
            getRequest.onerror = () => resolve(undefined);
            tx.oncomplete = () => db.close();
        } catch (e) {
            console.error("IndexedDB transaction error", e);
            resolve(undefined);
            if(db) db.close();
        }
      };
      request.onerror = () => resolve(undefined);
    });
  },
  set: <T>(key: string, value: T): Promise<void> => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('persistent-blob-store', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('blobs');
      request.onsuccess = () => {
        const db = request.result;
        try {
            const tx = db.transaction('blobs', 'readwrite');
            const store = tx.objectStore('blobs');
            const setRequest = store.put(value, key);
            setRequest.onsuccess = () => resolve();
            setRequest.onerror = (e) => reject(e);
            tx.oncomplete = () => db.close();
        } catch(e) {
            reject(e);
            if(db) db.close();
        }
      };
      request.onerror = (e) => reject(e);
    });
  },
  del: (key: string): Promise<void> => {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('persistent-blob-store', 1);
        request.onupgradeneeded = () => request.result.createObjectStore('blobs');
        request.onsuccess = () => {
            const db = request.result;
            try {
                const tx = db.transaction('blobs', 'readwrite');
                const store = tx.objectStore('blobs');
                const delRequest = store.delete(key);
                delRequest.onsuccess = () => resolve();
                delRequest.onerror = (e) => reject(e);
                tx.oncomplete = () => db.close();
            } catch(e) {
                reject(e);
                if(db) db.close();
            }
        };
        request.onerror = (e) => reject(e);
    });
  },
};


export function usePersistentBlob(key: string): [string | null, Blob | null, (blob: Blob | null) => void, boolean] {
    const [objectUrl, setObjectUrl] = useState<string | null>(null);
    const [blob, setBlobState] = useState<Blob | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        let currentUrl: string | null = null;
        let isActive = true;

        // Immediately reset state when the key changes to prevent showing stale data
        setObjectUrl(null);
        setBlobState(null);
        setIsLoading(true);

        idb.get<Blob>(key).then(loadedBlob => {
            if (isActive && loadedBlob) {
                setBlobState(loadedBlob);
                currentUrl = URL.createObjectURL(loadedBlob);
                setObjectUrl(currentUrl);
            }
        }).catch(err => {
            console.error(`Failed to get blob for key "${key}" from IndexedDB`, err);
        }).finally(() => {
            if (isActive) {
                setIsLoading(false);
            }
        });

        return () => {
            isActive = false;
            if (currentUrl) {
                URL.revokeObjectURL(currentUrl);
            }
        };
    }, [key]);

    const setAndStoreBlob = (newBlob: Blob | null) => {
        if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
        }

        setBlobState(newBlob);

        if (newBlob) {
            idb.set(key, newBlob).then(() => {
                const newUrl = URL.createObjectURL(newBlob);
                setObjectUrl(newUrl);
            }).catch(err => {
                console.error(`Failed to set blob for key "${key}" in IndexedDB`, err);
                setObjectUrl(null);
            });
        } else {
            idb.del(key).then(() => {
                setObjectUrl(null);
            }).catch(err => {
                console.error(`Failed to delete blob for key "${key}" in IndexedDB`, err);
            });
        }
    };
    
    return [objectUrl, blob, setAndStoreBlob, isLoading];
}