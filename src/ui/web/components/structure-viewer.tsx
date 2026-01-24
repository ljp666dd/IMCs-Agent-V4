import React, { useEffect, useRef } from 'react';

// Declare global 3Dmol for TypeScript since we might load it globally or via require
declare global {
    interface Window {
        $3Dmol: any;
    }
}

interface StructureViewerProps {
    cifContent: string;
    className?: string;
}

const StructureViewer: React.FC<StructureViewerProps> = ({ cifContent, className }) => {
    const viewerRef = useRef<HTMLDivElement>(null);
    const glRef = useRef<any>(null);

    useEffect(() => {
        // Dynamic import to avoid SSR issues
        const initViewer = async () => {
            if (!viewerRef.current) return;

            try {
                const $3Dmol = (await import('3dmol')).default; // If using npm package
                // Or if using CDN script, check window.$3Dmol

                const element = viewerRef.current;
                const config = { backgroundColor: 'white' };

                // Initialize viewer
                const viewer = $3Dmol.createViewer(element, config);
                glRef.current = viewer;

                // Add Model
                viewer.addModel(cifContent, "cif");
                viewer.setStyle({}, { sphere: { scale: 0.3 }, stick: { radius: 0.2 } }); // Ball and Stick
                viewer.zoomTo();
                viewer.render();

            } catch (e) {
                console.error("Failed to init 3Dmol", e);
            }
        };

        if (cifContent) {
            initViewer();
        }

        return () => {
            // Cleanup if possible (3Dmol doesn't have a clear dispose)
        };
    }, [cifContent]);

    return (
        <div
            ref={viewerRef}
            className={`border rounded-lg overflow-hidden bg-white shadow-inner relative w-full h-[400px] ${className}`}
        >
            {!cifContent && (
                <div className="absolute inset-0 flex items-center justify-center text-neutral-400">
                    No Structure Data
                </div>
            )}
        </div>
    );
};

export default StructureViewer;
