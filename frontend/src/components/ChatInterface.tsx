import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, Sparkles } from 'lucide-react';
import { Message, ChatConfig } from '../lib/types';
import { sendMessage } from '../lib/api';
import { MessageBubble } from './MessageBubble';
import { SettingsPanel } from './SettingsPanel';
import { cn } from '../lib/utils';

export const ChatInterface: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [config, setConfig] = useState<ChatConfig>({
        model: 'llama3',
        temperature: 0.0,
        max_tokens: 1000,
    });
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await sendMessage(input.trim(), config);

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: response.answer,
                timestamp: new Date(),
                sources: response.sources,
            };

            setMessages((prev) => [...prev, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
            inputRef.current?.focus();
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-screen relative overflow-hidden">
            {/* Animated Background Orbs */}
            <div className="floating-orb w-96 h-96 bg-liquid-primary -top-48 -left-48" style={{ animationDelay: '0s' }} />
            <div className="floating-orb w-80 h-80 bg-liquid-secondary top-1/3 -right-40" style={{ animationDelay: '2s' }} />
            <div className="floating-orb w-72 h-72 bg-liquid-accent -bottom-36 left-1/4" style={{ animationDelay: '4s' }} />

            {/* Header */}
            <motion.header
                initial={{ y: -100 }}
                animate={{ y: 0 }}
                className="glass-panel border-b border-glass-200 px-6 py-4 flex items-center justify-between relative z-10"
            >
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-liquid-primary to-liquid-secondary flex items-center justify-center">
                        <Sparkles className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-gradient">Liquid Glass AI</h1>
                        <p className="text-xs text-gray-400">RAG-Powered Chatbot</p>
                    </div>
                </div>
                <SettingsPanel config={config} onConfigChange={setConfig} />
            </motion.header>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto px-6 py-6 relative z-10">
                {messages.length === 0 ? (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="flex flex-col items-center justify-center h-full text-center"
                    >
                        <div className="glass-panel p-8 rounded-3xl max-w-md">
                            <Sparkles className="w-16 h-16 text-liquid-primary mx-auto mb-4" />
                            <h2 className="text-2xl font-bold text-gradient mb-2">
                                Welcome to Liquid Glass AI
                            </h2>
                            <p className="text-gray-400 text-sm">
                                Ask me anything. I use a 5-stage RAG pipeline with query
                                expansion, hybrid retrieval, reranking, compression, and
                                advanced generation.
                            </p>
                        </div>
                    </motion.div>
                ) : (
                    <div className="max-w-4xl mx-auto">
                        {messages.map((message) => (
                            <MessageBubble key={message.id} message={message} />
                        ))}
                        {isLoading && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="flex items-center gap-3 mb-4"
                            >
                                <div className="glass-panel w-10 h-10 rounded-full flex items-center justify-center">
                                    <Loader2 className="w-5 h-5 text-liquid-primary animate-spin" />
                                </div>
                                <div className="glass-panel px-4 py-3 rounded-2xl">
                                    <div className="flex gap-1">
                                        <span className="w-2 h-2 bg-liquid-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                        <span className="w-2 h-2 bg-liquid-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                        <span className="w-2 h-2 bg-liquid-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                    </div>
                                </div>
                            </motion.div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area */}
            <motion.div
                initial={{ y: 100 }}
                animate={{ y: 0 }}
                className="glass-panel border-t border-glass-200 px-6 py-4 relative z-10"
            >
                <div className="max-w-4xl mx-auto flex gap-3">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask me anything..."
                        disabled={isLoading}
                        className="liquid-input flex-1"
                    />
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className={cn(
                            'liquid-button px-6 flex items-center gap-2',
                            (isLoading || !input.trim()) && 'opacity-50 cursor-not-allowed'
                        )}
                    >
                        {isLoading ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Send className="w-5 h-5" />
                        )}
                        Send
                    </motion.button>
                </div>
            </motion.div>
        </div>
    );
};
