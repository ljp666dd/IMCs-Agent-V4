'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Brain, BarChart, Play, Activity } from "lucide-react";
import Link from 'next/link';
import { AgentService } from "@/lib/api";

export default function MLPage() {
    const [training, setTraining] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);

    const handleTrain = async (type: string) => {
        setTraining(true);
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Starting ${type} training...`]);
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Loading data from DB...`]);

        try {
            // Trigger backend task (Using Task Agent for now to orchestrate, or we could add direct /ml/train endpoint later if needed)
            // For v3.4, we use the Chat interface logic but triggered here via specific instruction
            const query = `Train ${type} models on current data`;
            setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Dispatching task: "${query}"`]);

            const plan = await AgentService.createTask(query);
            await AgentService.executeTask(plan.task_id);

            setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Task ${plan.task_id} started. Check logs for details.`]);
            setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training running in background...`]);

        } catch (e) {
            setLogs(prev => [...prev, `[ERROR] Failed to start training: ${e}`]);
        } finally {
            setTraining(false);
        }
    };

    return (
        <main className="min-h-screen bg-neutral-50 p-8">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center space-x-4">
                    <Link href="/" className="p-2 hover:bg-neutral-200 rounded-full transition-colors">
                        <ArrowLeft className="h-6 w-6 text-neutral-600" />
                    </Link>
                    <h1 className="text-2xl font-bold text-neutral-900">Machine Learning Center</h1>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                    {/* Left Col: Setup & Actions */}
                    <div className="lg:col-span-1 space-y-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Brain className="h-5 w-5 text-purple-600" /> Model Training
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Traditional ML</label>
                                    <Button
                                        onClick={() => handleTrain("traditional")}
                                        disabled={training}
                                        className="w-full justify-start variant-outline bg-white border hover:bg-neutral-50 text-neutral-800"
                                    >
                                        <Play className="h-4 w-4 mr-2 text-green-600" /> Train RF / XGBoost
                                    </Button>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Deep Learning</label>
                                    <Button
                                        onClick={() => handleTrain("deep learning")}
                                        disabled={training}
                                        className="w-full justify-start variant-outline bg-white border hover:bg-neutral-50 text-neutral-800"
                                    >
                                        <Play className="h-4 w-4 mr-2 text-blue-600" /> Train DNN / Transformer
                                    </Button>
                                </div>

                                <div className="pt-4 border-t">
                                    <p className="text-xs text-neutral-500">Data Source: <span className="font-semibold text-neutral-700">Project Database</span></p>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Activity className="h-5 w-5 text-orange-600" /> Live Logs
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="bg-neutral-900 text-green-400 font-mono text-xs p-4 rounded-b-lg h-[200px] overflow-y-auto">
                                {logs.length === 0 ? (
                                    <span className="text-neutral-600">Waiting for actions...</span>
                                ) : (
                                    logs.map((log, i) => <div key={i}>{log}</div>)
                                )}
                            </CardContent>
                        </Card>
                    </div>

                    {/* Right Col: Performance & SHAP */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Model Registry (Mocked logic for v3.4 until API /ml/models is hooked up completely) */}
                        <Card>
                            <CardHeader>
                                <CardTitle>Model Performance Registry</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="rounded-md border">
                                    <table className="w-full text-sm">
                                        <thead className="bg-neutral-50 border-b">
                                            <tr>
                                                <th className="h-10 px-4 text-left font-medium text-neutral-500">Model Name</th>
                                                <th className="h-10 px-4 text-left font-medium text-neutral-500">Type</th>
                                                <th className="h-10 px-4 text-left font-medium text-neutral-500">R² Score</th>
                                                <th className="h-10 px-4 text-left font-medium text-neutral-500">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr className="border-b">
                                                <td className="p-4 font-medium">XGBoost_v1</td>
                                                <td className="p-4">Gradient Boosting</td>
                                                <td className="p-4">0.87</td>
                                                <td className="p-4"><span className="px-2 py-1 rounded-full bg-green-100 text-green-700 text-xs">Ready</span></td>
                                            </tr>
                                            <tr className="border-b">
                                                <td className="p-4 font-medium">DNN_Large</td>
                                                <td className="p-4">Neural Network</td>
                                                <td className="p-4">0.92</td>
                                                <td className="p-4"><span className="px-2 py-1 rounded-full bg-green-100 text-green-700 text-xs">Ready</span></td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </CardContent>
                        </Card>

                        {/* SHAP Placeholder */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center justify-between">
                                    <span>Feature Importance (SHAP)</span>
                                    <BarChart className="h-4 w-4 text-neutral-500" />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="h-[250px] bg-neutral-100 rounded flex items-center justify-center text-neutral-400">
                                    Run training to generate feature analysis
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </main>
    );
}
