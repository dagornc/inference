import React from 'react';
import { motion } from 'framer-motion';
import { Message } from '../lib/types';
import { cn } from '../lib/utils';
import { User, Bot } from 'lucide-react';

interface MessageBubbleProps {
    message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
    const isUser = message.role === 'user';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={cn(
                'flex gap-3 mb-4',
                isUser ? 'flex-row-reverse' : 'flex-row'
            )}
        >
            {/* Avatar */}
            <div
                className={cn(
                    'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
                    isUser
                        ? 'bg-gradient-to-br from-liquid-primary to-liquid-secondary'
                        : 'glass-panel'
                )}
            >
                {isUser ? (
                    <User className="w-5 h-5 text-white" />
                ) : (
                    <Bot className="w-5 h-5 text-liquid-primary" />
                )}
            </div>

            {/* Message Content */}
            <div
                className={cn(
                    'flex flex-col gap-2 max-w-[75%]',
                    isUser ? 'items-end' : 'items-start'
                )}
            >
                <div
                    className={cn(
                        'px-4 py-3 rounded-2xl',
                        isUser
                            ? 'message-bubble-user'
                            : 'message-bubble-bot'
                    )}
                >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {message.content}
                    </p>
                </div>

                {/* Sources (for bot messages) */}
                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-1">
                        {message.sources.slice(0, 3).map((source, idx) => (
                            <div
                                key={source.id}
                                className="glass-panel px-2 py-1 text-xs text-gray-300"
                            >
                                Source {idx + 1} ({(source.score * 100).toFixed(0)}%)
                            </div>
                        ))}
                    </div>
                )}

                {/* Timestamp */}
                <span className="text-xs text-gray-500 px-2">
                    {message.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                    })}
                </span>
            </div>
        </motion.div>
    );
};
