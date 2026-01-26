<template>
  <div class="textarea-content">
    <el-input
      v-model="chatData.content"
      @keydown="handleKeydown"
      :placeholder="
        generateTitle('在此输入您的问题或需求，有问必答，Shift+Enter换行')
      "
      type="textarea"
      :autosize="{ minRows: props.minRows }"
      :disabled="disableStatus"
    >
    </el-input>
    <div class="btn-box">
      <el-button type="primary" @click="submitFun" :disabled="disableStatus"
        ><el-icon color="#fff" size="16"><Promotion /></el-icon
        >{{ generateTitle('发送') }}</el-button
      >
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, defineProps, defineEmits, defineExpose } from 'vue'
import { Promotion } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { generateTitle } from '../utils/i18n'

// import { chat } from "../api/user";
const handleKeydown = (event: any) => {
  if (!event.shiftKey && event.keyCode == 13) {
    event.cancelBubble = true
    event.stopPropagation()
    event.preventDefault()
  }
}
const props = defineProps({
  minRows: {
    type: Number,
    default: 6
  }
})

const chatData = ref<any>({
  // chat_id: '',
  source: 'user',
  content: ''
})
//如何在室温条件下合成CsPbBr3
const disableStatus = ref(false)
const emits = defineEmits(['submitFun'])
const submitFun = async () => {
  if (!chatData.value.content) {
    ElMessage.error('请输入问题')
    return
  }
  if (disableStatus.value) {
    ElMessage.error('回答输出中，暂不能再次提问')
    return
  }
  // chatData.value.chat_id = new Date().getTime()
  emits('submitFun', JSON.stringify(chatData.value))
  chatData.value = {
    // chat_id: '',
    source: 'user',
    content: ''
  }
}
defineExpose({
  disableStatus
})
</script>
<style scoped lang="less">
.textarea-content {
  position: relative;
  .btn-box {
    position: absolute;
    bottom: 12px;
    right: 12px;
    .el-button {
      display: flex;
      align-items: center;
      justify-content: center;
      column-gap: 4px;
    }
  }
  :deep(.el-textarea__inner) {
    width: 100%;
    border-radius: 6px;
    background: rgba(0, 30, 70, 0.5);
    border: 1px solid rgba(59, 164, 255, 1);
    box-shadow: 0px 0px 16px 0px rgba(0, 149, 230, 1) inset;
    padding: 16px;
    font-size: 14px;
    color: rgba(157, 215, 248, 1);
    box-sizing: border-box;

    &::-webkit-scrollbar {
      width: 0;
      height: 0;
    }

    &:focus {
      outline: 1px solid #3ba4ff;
    }

    &::-webkit-input-placeholder {
      color: rgba(157, 215, 248, 1);
    }

    &:-moz-placeholder {
      color: rgba(157, 215, 248, 1);
    }

    &::-moz-placeholder {
      color: rgba(157, 215, 248, 1);
    }

    &::-ms-input-placeholder {
      color: rgba(157, 215, 248, 1);
    }
  }
}
</style>
