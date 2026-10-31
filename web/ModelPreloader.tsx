'use client';

import { useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

/**
 * Module-level singleton for the pre-loaded ASL model.
 * Accessible from any component via getPreloadedModel().
 */
let preloadedModel: tf.GraphModel | null = null;
let preloadPromise: Promise<tf.GraphModel | null> | null = null;

const INDEXEDDB_MODEL_KEY = 'indexeddb://asl-model';
const NETWORK_MODEL_URL_SUFFIX = '/models/asl/model.json';

/**
 * Get the pre-loaded model. Returns immediately if already loaded,
 * or waits for the in-progress preload to finish.
 */
export const getPreloadedModel = async (): Promise<tf.GraphModel | null> => {
    if (preloadedModel) return preloadedModel;
    if (preloadPromise) return preloadPromise;
    // Edge case: preloader hasn't started yet, trigger it
    preloadPromise = loadModel();
    return preloadPromise;
};

/**
 * Check if the model is already loaded (synchronous check).
 */
export const isModelReady = (): boolean => {
    return preloadedModel !== null;
};

/**
 * Core model loading logic:
 * 1. Set WebGL backend
 * 2. Try loading from IndexedDB cache
 * 3. Fall back to network fetch + cache to IndexedDB
 */
async function loadModel(): Promise<tf.GraphModel | null> {
    try {
        // Prefer WebGL, but keep the demo usable on browsers without WebGL.
        try {
            await tf.setBackend('webgl');
        } catch {
            await tf.setBackend('cpu');
        }
        await tf.ready();
        console.log('[ModelPreloader] TFJS backend:', tf.getBackend());

        // Check if model is cached in IndexedDB
        const savedModels = await tf.io.listModels();
        if (savedModels[INDEXEDDB_MODEL_KEY]) {
            console.log('[ModelPreloader] Loading model from IndexedDB cache...');
            try {
                const model = await tf.loadGraphModel(INDEXEDDB_MODEL_KEY);
                preloadedModel = model;
                console.log('[ModelPreloader] Model loaded from IndexedDB cache successfully.');
                return model;
            } catch (cacheErr) {
                console.warn('[ModelPreloader] IndexedDB cache corrupted, fetching from network...', cacheErr);
                // Remove corrupted cache
                await tf.io.removeModel(INDEXEDDB_MODEL_KEY);
            }
        }

        // Fetch from network
        const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
        const modelUrl = `${baseUrl}${NETWORK_MODEL_URL_SUFFIX}`;
        console.log('[ModelPreloader] Downloading model from network...');
        const model = await tf.loadGraphModel(modelUrl);
        preloadedModel = model;

        // Cache to IndexedDB for next visit
        try {
            await model.save(INDEXEDDB_MODEL_KEY);
            console.log('[ModelPreloader] Model cached to IndexedDB for future visits.');
        } catch (saveErr) {
            console.warn('[ModelPreloader] Failed to cache model to IndexedDB:', saveErr);
        }

        console.log('[ModelPreloader] Model loaded from network successfully.');
        return model;
    } catch (err) {
        console.error('[ModelPreloader] Failed to preload model:', err);
        return null;
    }
}

/**
 * Invisible component that triggers model preloading on mount.
 * Place in layout.tsx to start loading as soon as the page opens.
 */
const ModelPreloader = () => {
    useEffect(() => {
        if (!preloadPromise) {
            preloadPromise = loadModel();
        }
    }, []);

    return null; // Renders nothing
};

export default ModelPreloader;
