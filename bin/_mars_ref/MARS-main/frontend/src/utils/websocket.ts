function useWebsocket(handleMessage: any,url: any) {
    const ws = new WebSocket(url||import.meta.env.VITE_WB_BASE_URL)
    const init = () => {
        bindEvent();
    }
    const bindEvent = () => {
        ws.addEventListener('open',handleOpen,false)
        ws.addEventListener('close',handleClose,false)
        ws.addEventListener('error',handleError,false)
        ws.addEventListener('message',handleMessage,false)
    }
    const handleOpen = (e: any) => {
        console.log('websocket open',e);
    }
    const handleClose = (e: any) => {
        console.log('websocket close',e);
    }
    const handleError = (e: any) => {
        console.log('webscoket error',e)
    }
    init();
    return ws
}

export default useWebsocket