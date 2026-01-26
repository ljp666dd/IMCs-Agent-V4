<template>
  <div class="camera_box">
    <div class="header_box">
      <el-icon color="#fff" size="20" @click="closeCamera"
        ><CircleClose
      /></el-icon>
    </div>
    <div class="img_box">
      <!-- <el-icon class="icon_box" color="#fff" size="36" v-show="!loading"
        ><Loading
      /></el-icon> -->
      <img
        :src="imgOneVal"
        style="width: 50vw; height: 50vh"
        alt=""
        v-show="cameraNumber === 1"
      />
      <img
        :src="imgTwoVal"
        style="width: 50vw; height: 50vh"
        alt=""
        v-show="cameraNumber === 2"
      />
    </div>
    <!-- <canvas ref="cameraOne" style="width: 50vw; height: 50vh"></canvas> -->
    <!-- <canvas ref="cameraTwo" style="width: 50vw; height: 50vh"></canvas> -->
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, defineExpose, defineEmits, watch} from 'vue'
import useWebSocket from '../../utils/websocket'
// const cameraOne = ref("")
// const cameraTwo = ref("");
const imgOneVal = ref('')
const imgTwoVal = ref('')
const loading = ref<boolean>(false)
const handleMessage = (e: any) => {
  imgOneVal.value = 'data:image/jpeg;base64,' + e.data
  if (!loading.value && imgOneVal.value) {
    loading.value = true
  }
}
const handleTwoMessage = (e: any) => {
  imgTwoVal.value = 'data:image/jpeg;base64,' + e.data
  if (!loading.value && imgTwoVal.value) {
    loading.value = true
  }
}
const cameraNumber = ref<Number>(0)

const ws = ref<any>(null) // WebSocket instance for camera 1
const wsTwo = ref<any>(null) // WebSocket instance for camera 2
// const ws: any = useWebSocket(handleMessage, import.meta.env.VITE_WB_CAMERA_ONE_URL)
// const wsTwo: any = useWebSocket(
//   handleTwoMessage,
//   import.meta.env.VITE_WB_CAMERA_TWO_URL
// )
const emit = defineEmits(['closeCamera'])
// const ws = new WebSocket(import.meta.env.VITE_WB_CAMERA_ONE_URL)


const stopWebSocket = () =>{
  // Close WebSocket connections
  if (ws.value) {
    ws.value.close()
    ws.value = null
  }
  if (wsTwo.value) {
    wsTwo.value.close()
    wsTwo.value = null
  }
}
const closeCamera = () => {
  // Reset the camera number
  cameraNumber.value = 0
  stopWebSocket() // Stop any active WebSocket connection
  emit('closeCamera')
}
watch(cameraNumber, (newVal) => {
  stopWebSocket()

  if(newVal == 1){
    ws.value = useWebSocket(handleMessage, import.meta.env.VITE_WB_CAMERA_ONE_URL)
  }else if(newVal == 2){
    wsTwo.value = useWebSocket(handleTwoMessage, import.meta.env.VITE_WB_CAMERA_TWO_URL)
  }
})
// })
// const closeWs = () => {
//   if (ws) {
//     ws.handleClose()
//   }
// }
// const closeWsTwo = () => {
//   if (wsTwo) {
//     wsTwo.handleClose()
//   }
// }
// onMounted(() => {})

onUnmounted(() => {
  stopWebSocket()
})

defineExpose({ cameraNumber })

</script>

<style scoped>
@import './index.less';
</style>
