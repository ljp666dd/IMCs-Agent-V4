<template>
  <div class="reasoning-content">
    <div
      class="content-left"
      :style="{
        width: isCollapse ? 'calc(100% - 200px)' : 'calc(100% - 7px)'
      }"
    >
      <div class="body-box" ref="container">
        <reasoning-view
          ref="reasoningRef"
          @completeFun="completeFun"
          :reasoningList="reasoningList"
        />
      </div>
      <div class="content-box">
        <div class="tip-box">
          <p class="active_item">
            <el-icon size="6" color="#fff"><Plus /></el-icon>
            {{ generateTitle('新建对话') }}
          </p>
          <div class="tip-text" v-show="disableStatus">
            {{ generateTitle('回答输出中，暂不能再次提问') }}
          </div>
        </div>
        <TextareaView ref="textareaRef" :minRows="1" @submitFun="submitFun" />
      </div>
    </div>
    <collapseView
      ref="collapseRef"
      :reasoningList="reasoningList"
      @changeStatusFun="changeStatusFun"
    />
    <div class="camera_dialog">
      <div class="camera_item" @click="cameraFun(1)">
        <el-icon color="#fff" size="16"><VideoCameraFilled /></el-icon>
        <span>{{ generateTitle('监控1') }}</span>
      </div>
      <div class="camera_item" @click="cameraFun(2)">
        <el-icon color="#fff" size="16"><VideoCameraFilled /></el-icon>
        <span>{{ generateTitle('监控2') }}</span>
      </div>
    </div>
    <!-- <div class="message_btn">
      <el-icon color="#fff" size="16" v-show="messageShow"><Expand /></el-icon>
      <el-icon color="#fff" size="16" v-show="!messageShow"><Fold /></el-icon>
    </div> -->
    <el-drawer
      v-model="messageShow"
      title="I am the title"
      direction="ltr"
      :before-close="handleClose"
    >
      <span>Hi, there!</span>
    </el-drawer>
    <CameraView
      ref="cameraViewRef"
      v-show="cameraShow"
      @closeCamera="closeCamera"
    />
  </div>
</template>
<script setup lang="ts">
import ReasoningView from '../../components/ReasoningView/index.vue'
import TextareaView from '../../components/TextareaView.vue'
import collapseView from '../../components/collapseView/index.vue'
import CameraView from '../../components/CameraView/index.vue'
// Expand, Fold
import { Plus, VideoCameraFilled } from '@element-plus/icons-vue'
import { ref, onMounted, nextTick } from 'vue'
// import { getModelList } from '../../api/user'
import { useRoute } from 'vue-router'
import useWebSocket from '../../utils/websocket'
// import { getAgent } from '../../utils/agent'
import { generateTitle } from '../../utils/i18n'
const route: any = useRoute()
const reasoningRef = ref<any>(null)
const textareaRef = ref<any>(null)
const collapseRef = ref<any>(null)
const isCollapse = ref<boolean>(true)
const cameraShow = ref<boolean>(false)
const messageShow = ref<boolean>(false)
const cameraViewRef = ref<any>(null)
const changeStatusFun = (val: boolean) => {
  isCollapse.value = val
}
const handleClose = () => {
  messageShow.value = false
}
const cameraFun = (val: number) => {
  cameraShow.value = true
  cameraViewRef.value.cameraNumber = val
}
const closeCamera = () => {
  cameraShow.value = false
}
const reasoningList = ref<Array<any>>([])
// const getModelListFun = async () => {
//   try {
//     const { data, code } = await getModelList()
//     if (code) {
//       reasoningList.value = data
//     }
//   } catch (error) {
//     console.log(error)
//   }
// }
const submitFun = (val: any) => {
  completeFun()
  // addMode(JSON.parse(val).content)
  ws.send(val)
}
const addMode = (val: any) => {
  endStatus.value = false
  reasoningList.value.push({
    title: val,
    children: []
  })
  reasoningRef.value.completeList.push({
    show: false,
    index: reasoningRef.value.completeList.length
  })
}
const disableStatus = ref(false)
const completeFun = () => {
  disableStatus.value = true
  textareaRef.value.disableStatus = true
}
const handleMessage = (e: any) => {
  getMessage(e.data)
  getHeight()
}
const endStatus = ref(false)
const getMessage = async (e: any) => {
  reasoningRef.value.reasonStatus.show = true
  let data = JSON.parse(e)
  if (data.type === 'UserInputRequestedEvent') {
    disableStatus.value = false
    textareaRef.value.disableStatus = false
    reasoningRef.value.reasonStatus.show = false
  }
  if (data.content) {
    let list = reasoningList.value[reasoningList.value.length - 1].children
    list.push(data)
  }
}
const ws = useWebSocket(handleMessage, '')
const sendFun = () => {
  setTimeout(() => {
    addMode(JSON.parse(route.query.content).content)
    completeFun()
    ws.send(route.query.content)
  }, 300)
}
const container = ref<any>(null)
const getHeight = () => {
  container.value.scrollTop = container.value.scrollHeight + 150
  collapseRef.value.scrollTop = collapseRef.value.scrollHeight + 50
}
onMounted(() => {
  // getModelListFun();
  nextTick(() => {
    getHeight()
    sendFun()
  })
})
</script>
<style scoped>
@import './index.less';
</style>
