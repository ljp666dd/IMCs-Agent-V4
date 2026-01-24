'use client';

import { useState, useEffect, useRef } from 'react';
import { AgentService, TaskResponse } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Bot, User, Send, PlayCircle, CheckCircle } from "lucide-react";
import Link from 'next/link';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    plan?: TaskResponse;
}

export default function ChatPage() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([
        { role: 'assistant', content: 'Hello! I am your research assistant. How can I help you today? (e.g., "Find HER catalysts" or "Train ML model")' }
    ]);
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setLoading(true);

        try {
            // 1. Send Message to Unified Chat Endpoint
            const res = await AgentService.chat(userMsg);

            if (res.type === 'plan') {
                const plan = res.content;
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `I have created a plan for: "${plan.description}". You can review it below.`,
                    plan: plan
                }]);
            } else {
                // Regular Chat Response
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: res.content || JSON.stringify(res)
                }]);
            }
        } catch (e) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error connecting to the backend.' }]);
        } finally {
            setLoading(false);
        }
    };

    const [executing, setExecuting] = useState<string | null>(null);

    const handleExecute = async (taskId: string) => {
        setExecuting(taskId);
        try {
            await AgentService.executeTask(taskId);

            // Poll for completion
            const interval = setInterval(async () => {
                try {
                    const statusRes = await AgentService.getTaskStatus(taskId);
                    if (statusRes.status === 'completed') {
                        clearInterval(interval);
                        setExecuting(null);

                        // Find recommendation in results
                        let recommendation = "Task completed. Check logs for details.";
                        if (statusRes.results) {
                            // Search for step with recommendation
                            for (const key in statusRes.results) {
                                const stepRes = statusRes.results[key];
                                if (stepRes && stepRes.recommendation) {
                                    recommendation = stepRes.recommendation;
                                    break;
                                }
                            }
                        }

                        setMessages(prev => [...prev, {
                            role: 'assistant',
                            content: recommendation
                        }]);
                    } else if (statusRes.status === 'failed') {
                        clearInterval(interval);
                        setExecuting(null);
                        alert("Task execution failed.");
                    }
                } catch (e) {
                    console.error("Polling error", e);
                }
            }, 3000);

        } catch (e) {
            setExecuting(null);
            alert("Failed to start execution");
        }
    };

    return (
        <main className="min-h-screen bg-neutral-50 p-4 md:p-8 flex flex-col">
            {/* Header */}
            <div className="flex items-center space-x-4 mb-6 max-w-4xl mx-auto w-full">
                <Link href="/" className="p-2 hover:bg-neutral-200 rounded-full transition-colors">
                    <ArrowLeft className="h-6 w-6 text-neutral-600" />
                </Link>
                <h1 className="text-2xl font-bold text-neutral-900">Research Assistant</h1>
            </div>

            {/* Chat Container */}
            <Card className="flex-1 max-w-4xl mx-auto w-full flex flex-col overflow-hidden max-h-[80vh]">
                <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`flex max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'} items-start gap-3`}>

                                {/* Avatar */}
                                <div className={`p-2 rounded-full ${msg.role === 'user' ? 'bg-blue-600' : 'bg-green-600'} text-white shrink-0`}>
                                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                                </div>

                                {/* Bubble */}
                                <div className={`p-3 rounded-lg text-sm ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border shadow-sm text-neutral-800'}`}>
                                    <p>{msg.content}</p>

                                    {/* Plan Card if available */}
                                    {msg.plan && (
                                        <div className="mt-3 bg-neutral-50 border rounded p-3 text-neutral-900">
                                            <div className="font-semibold border-b pb-1 mb-2 flex justify-between items-center">
                                                <span>📋 Execution Plan</span>
                                                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">{msg.plan.task_type}</span>
                                            </div>
                                            <ul className="space-y-1 text-xs mb-3">
                                                {/* Task steps are simple dicts according to API */}
                                                {(msg.plan as any).steps?.map((step: any, i: number) => (
                                                    <li key={i} className="flex items-start gap-2">
                                                        <span className="bg-neutral-200 px-1.5 rounded">{i + 1}</span>
                                                        <span>[{step.agent?.toUpperCase()}] {step.action}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                            <Button
                                                size="sm"
                                                onClick={() => handleExecute(msg.plan!.task_id)}
                                                className="w-full bg-green-600 hover:bg-green-700"
                                            >
                                                <PlayCircle className="w-4 h-4 mr-2" /> Execute
                                            </Button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-white border shadow-sm p-3 rounded-lg text-neutral-500 text-sm animate-pulse">
                                Thinking...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </CardContent>

                {/* Input Area */}
                <div className="p-4 bg-white border-t flex gap-2">
                    <Input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask me to find materials or analyze data..."
                        className="flex-1"
                        disabled={loading}
                    />
                    <Button onClick={handleSend} disabled={loading || !input.trim()}>
                        <Send className="w-4 h-4" />
                    </Button>
                </div>
            </Card>
        </main>
    );
}
