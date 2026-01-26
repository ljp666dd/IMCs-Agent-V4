<template>
  <div class="content">
    <div class="title-box">
      <img src="../../assets/logo.png" alt="" />
      <span>{{ generateTitle('您好，多智能体MARS为您服务') }}</span>
    </div>
    <TextareaView @submitFun="reasoningFun" :minRows="6"></TextareaView>
    <div class="recommend-box">
      <div class="recommend-title">
        <img src="../../assets/home/recommend.png" alt="" />
        <span>{{ generateTitle('推荐问题') }}</span>
      </div>
      <div class="recommend-bottm">
        <div class="recommend-list">
          <div
            class="item"
            @click="submitFun(item)"
            v-for="(item, index) in recommendList"
            :key="index"
          >
            {{ item }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import TextareaView from '../../components/TextareaView.vue'
import { generateTitle } from '../../utils/i18n'

const recommendList = ref<Array<any>>([
  'Structure: How can the self-assembly of materials be achieved for nanoparticles by controlling intermolecular interactions?',
  'Property: For perovskite nanocrystals, how can blue-shifting of the fluorescence emission wavelength be controlled through structural design?',
  'Synthesis: How to synthesize CsPbBr₃ nanocubes at room temperature?',
  'Application: How can the photoluminescence quantum efficiency of perovskites in LED devices be improved?'
])
const router = useRouter()
const submitFun = (e: any) => {
  let data = {
    // chat_id: new Date().getTime(),
    // message: e
    content: e,
    source: 'user'
  }
  router.push('/reasoning?content=' + JSON.stringify(data))
}
const reasoningFun = (e: any) => {
  router.push('/reasoning?content=' + e)
}
onMounted(() => {})
</script>
<style scoped>
@import './index.less';
</style>
