import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, X, Sliders } from 'lucide-react';
import { ChatConfig } from '../lib/types';
import { cn } from '../lib/utils';

interface SettingsPanelProps {
    config: ChatConfig;
    onConfigChange: (config: ChatConfig) => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
    config,
    onConfigChange,
}) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <>
            {/* Settings Button */}
            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsOpen(true)}
                className="glass-panel-hover p-3 rounded-xl"
            >
                <Settings className="w-5 h-5 text-liquid-primary" />
            </motion.button>

            {/* Settings Panel Overlay */}
            <AnimatePresence>
                {isOpen && (
                    <>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsOpen(false)}
                            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
                        />

                        {/* Panel */}
                        <motion.div
                            initial={{ x: '100%' }}
                            animate={{ x: 0 }}
                            exit={{ x: '100%' }}
                            transition={{ type: 'spring', damping: 25 }}
                            className="fixed right-0 top-0 h-full w-96 glass-panel border-l border-glass-300 z-50 p-6 overflow-y-auto"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-2">
                                    <Sliders className="w-5 h-5 text-liquid-primary" />
                                    <h2 className="text-xl font-bold text-gradient">Settings</h2>
                                </div>
                                <button
                                    onClick={() => setIsOpen(false)}
                                    className="glass-panel-hover p-2 rounded-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Settings Form */}
                            <div className="space-y-6">
                                {/* Model Selection */}
                                <div>
                                    <label className="block text-sm font-medium mb-2 text-gray-300">
                                        Model
                                    </label>
                                    <input
                                        type="text"
                                        value={config.model || 'llama3'}
                                        onChange={(e) =>
                                            onConfigChange({ ...config, model: e.target.value })
                                        }
                                        className="liquid-input"
                                        placeholder="e.g., llama3, gpt-4"
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Default: llama3 (Ollama)
                                    </p>
                                </div>

                                {/* Temperature */}
                                <div>
                                    <label className="block text-sm font-medium mb-2 text-gray-300">
                                        Temperature: {config.temperature?.toFixed(1) || '0.0'}
                                    </label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.temperature || 0}
                                        onChange={(e) =>
                                            onConfigChange({
                                                ...config,
                                                temperature: parseFloat(e.target.value),
                                            })
                                        }
                                        className="w-full accent-liquid-primary"
                                    />
                                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                                        <span>Precise</span>
                                        <span>Creative</span>
                                    </div>
                                </div>

                                {/* Max Tokens */}
                                <div>
                                    <label className="block text-sm font-medium mb-2 text-gray-300">
                                        Max Tokens
                                    </label>
                                    <input
                                        type="number"
                                        value={config.max_tokens || 1000}
                                        onChange={(e) =>
                                            onConfigChange({
                                                ...config,
                                                max_tokens: parseInt(e.target.value),
                                            })
                                        }
                                        className="liquid-input"
                                        min="100"
                                        max="4000"
                                        step="100"
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Maximum response length
                                    </p>
                                </div>

                                {/* Info Box */}
                                <div className="glass-panel p-4 rounded-xl border border-liquid-primary/30">
                                    <h3 className="text-sm font-semibold mb-2 text-liquid-primary">
                                        Pipeline Info
                                    </h3>
                                    <ul className="text-xs text-gray-400 space-y-1">
                                        <li>✓ Query Expansion</li>
                                        <li>✓ Hybrid Retrieval</li>
                                        <li>✓ Multi-Stage Reranking</li>
                                        <li>✓ Contextual Compression</li>
                                        <li>✓ Advanced Generation</li>
                                    </ul>
                                </div>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </>
    );
};
