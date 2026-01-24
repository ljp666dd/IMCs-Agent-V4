'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, XCircle, Loader2, Circle, ArrowDown, Bot, Database, Beaker, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface TaskStep {
    step_id: string;
    agent: string;
    action: string;
    status: string;
    params?: Record<string, any>;
    dependencies: string[];
    result?: any;
    error?: string;
}

interface TaskPlan {
    task_id: string;
    task_type: string;
    description: string;
    steps: TaskStep[];
    status: string;
}

const AgentIcon = ({ agent }: { agent: string }) => {
    switch (agent.toLowerCase()) {
        case 'literature': return <FileText className="h-5 w-5 text-amber-600" />;
        case 'theory': return <AtomIcon />;
        case 'ml': return <Bot className="h-5 w-5 text-purple-600" />;
        case 'experiment': return <Beaker className="h-5 w-5 text-green-600" />;
        default: return <Database className="h-5 w-5 text-blue-600" />;
    }
};

const AtomIcon = ({ className }: { className?: string }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={`text-blue-500 ${className}`}
    >
        <circle cx="12" cy="12" r="1" />
        <path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5z" />
        <path d="M15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5z" />
    </svg>
);

const StatusIcon = ({ status }: { status: string }) => {
    switch (status.toLowerCase()) {
        case 'completed': return <CheckCircle className="h-5 w-5 text-green-500" />;
        case 'failed': return <XCircle className="h-5 w-5 text-red-500" />;
        case 'running': return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
        default: return <Circle className="h-5 w-5 text-neutral-300" />;
    }
};

export default function TaskGraphViewer({ plan }: { plan: TaskPlan }) {
    if (!plan || !plan.steps) return null;

    return (
        <Card className="w-full border-2 border-neutral-200 shadow-sm">
            <CardHeader className="pb-3 border-b bg-neutral-50 flex flex-row justify-between items-center">
                <div>
                    <CardTitle className="text-base font-semibold">Workflow Graph</CardTitle>
                    <p className="text-xs text-neutral-500 mt-1">{plan.description}</p>
                </div>
                <Badge variant="outline" className="uppercase text-xs tracking-wider">
                    {plan.status}
                </Badge>
            </CardHeader>
            <CardContent className="p-6 bg-neutral-50/50">
                <div className="space-y-4">
                    {plan.steps.map((step, idx) => (
                        <div key={step.step_id} className="relative group">
                            {/* Connector Line */}
                            {idx < plan.steps.length - 1 && (
                                <div className="absolute left-[1.65rem] top-10 bottom-[-1.5rem] w-0.5 bg-neutral-200 z-0" />
                            )}

                            <div className="flex items-start gap-4 relative z-10">
                                {/* Icon Bubble */}
                                <div className={`p-3 rounded-xl border shadow-sm transition-all duration-300 ${step.status === 'running' ? 'bg-blue-50 border-blue-200 scale-110' :
                                    step.status === 'completed' ? 'bg-white border-green-200' : 'bg-white border-neutral-200'
                                    }`}>
                                    <AgentIcon agent={step.agent} />
                                </div>

                                {/* Content Card */}
                                <div className={`flex-1 p-3 rounded-lg border bg-white ${step.status === 'running' ? 'border-blue-200 shadow-md transform translate-x-1' : 'border-neutral-200'
                                    } transition-all`}>
                                    <div className="flex justify-between items-start mb-1">
                                        <div className="flex items-center gap-2">
                                            <span className="font-semibold text-sm uppercase text-neutral-500 tracking-wide text-[10px]">
                                                {step.agent}
                                            </span>
                                            {step.status === 'completed' && <span className="text-[10px] text-green-600 bg-green-50 px-1.5 rounded">DONE</span>}
                                        </div>
                                        <StatusIcon status={step.status} />
                                    </div>

                                    <h4 className="font-medium text-neutral-900">{step.action}</h4>

                                    {/* Params (Collapsible?) */}
                                    <div className="text-xs text-neutral-500 mt-1 truncate">
                                        {Object.entries(step.params || {}).map(([k, v]) => `${k}=${v}`).join(', ')}
                                    </div>

                                    {/* Link to previous steps (Dependencies) */}
                                    {step.dependencies && step.dependencies.length > 0 && (
                                        <div className="mt-2 flex gap-1">
                                            {step.dependencies.map(dep => (
                                                <span key={dep} className="text-[10px] bg-neutral-100 text-neutral-500 px-1.5 py-0.5 rounded border">
                                                    Dep: {dep}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {/* Result Preview */}
                                    {step.result && (
                                        <div className="mt-2 p-2 bg-neutral-50 rounded border border-neutral-100 text-xs font-mono text-neutral-600 max-h-20 overflow-y-auto">
                                            {JSON.stringify(step.result).slice(0, 100)}...
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
