<template>
  <div class="login-box">
    <div v-loading="loading" class="content">
      <div class="title">
        <img src="/logo.png" alt="" />
        <span>{{generateTitle('MARS多智能体材料创制系统')}}</span>
      </div>
      <el-form :model="loginData">
        <el-form-item>
          <el-input
            :placeholder="generateTitle('账号')"
            v-model="loginData.user_name"
            :prefix-icon="UserFilled"
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-input
            :placeholder="generateTitle('密码')"
            type="password"
            v-model="loginData.pass_word"
            :prefix-icon="Lock"
            show-password
          ></el-input>
        </el-form-item>
        <el-form-item>
          <div class="code-box">
            <el-input
              :placeholder="generateTitle('验证码')"
              v-model="loginData.code"
              :prefix-icon="Lock"
            ></el-input>
            <ValidCode ref="validCodeRef" :height="56"></ValidCode>
          </div>
        </el-form-item>
      </el-form>
      <div class="btn-box">
        <el-button
          @click="submitFun"
          type="primary"
          element-loading-spinner="el-icon-loading"
          element-loading-background="rgba(0, 0, 0, 0.3)"
          v-loading.fullscreen.lock="loading"
          >{{generateTitle("登录")}}</el-button
        >
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { UserFilled, Lock } from '@element-plus/icons-vue'
import ValidCode from '../../components/ValidCode.vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { authLogin } from '../../api/user'
import {generateTitle} from '../../utils/i18n'
const router = useRouter()
const loginData = ref<any>({
  user_name: '',
  pass_word: '',
  code: ''
})
const loading = ref(false)
const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === 'Enter') {
    submitFun()
  }
}
const validCodeRef = ref()
const submitFun = async () => {
  if (!loginData.value.user_name) {
    ElMessage.error(generateTitle('请输入账号'))
    return
  }
  if (!loginData.value.pass_word) {
    ElMessage.error(generateTitle('请输入密码'))
    return
  }
  if (!loginData.value.user_name) {
    ElMessage.error(generateTitle('请输入验证码'))
    return
  }
  let code = validCodeRef.value.validate(loginData.value.code)
  if (!code) {
    ElMessage.error(generateTitle('验证码错误'))
    return
  }
  loading.value = true
  try {
    const { token }: any = await authLogin(loginData.value)
    localStorage.setItem('token', token)
    loading.value = false
    router.push('/home')
  } catch (error: any) {
    loading.value = false
    ElMessage.error(error.response.data.error)
  }
}
onMounted(() => {
  window.addEventListener('keypress', handleKeyPress)
})

onUnmounted(() => {
  window.removeEventListener('keypress', handleKeyPress)
})
</script>
<style scoped>
@import './index.less';
</style>
