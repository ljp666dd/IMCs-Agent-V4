<template>
  <div
    class="aside-box"
    :class="{ 'is-sidebar-open': isCollapse }"
    width="230px"
  >
    <div>
      <div
        class="collapse-demo"
        :class="['collapse-demo', { active: isCollapse }]"
        width="230px"
      >
        <div class="collapse-title">
          <div class="icon-box" @click="changeStatusFun">
            <el-icon size="14" color="rgba(157, 215, 248, 1)"
              ><component :is="getIcon()"></component
            ></el-icon>
          </div>
          <div class="title">{{ generateTitle('模型调用流程') }}</div>
        </div>
        <div style="height: 100%" v-show="isCollapse">
          <steps-view :reasoningList="reasoningList" />
        </div>
        <div v-show="isCollapse">
          <ProcessView />
        </div>
        <!-- <div class="user-box">
        <img src="../../assets/logo.png" alt="" />
        <span v-show="isCollapse">Jayson</span>
      </div> -->
        <!-- <div class="chart-box" v-show="isCollapse">
          <div class="body-box">
            <div class="out-round-list">
              <div
                class="out-round-item"
                v-for="(item, index) in chartList.slice(1, 6)"
                :key="index"
              >
                <div class="round-arrow-box">
                  <div class="round-lage-arrow">
                    <img src="../../assets/chart/round-lage-arrow.png" alt="" />
                  </div>
                  <div class="round-arrow" v-if="index != 4">
                    <img src="../../assets/chart/round-arrow.png" alt="" />
                  </div>
                </div>
                <div class="model-box" :class="{'chart-active': getStatus(item.group_name)}">
                <div
                  class="chart-bg"
                  v-if="getStatus(item.group_name)"
                ></div>
                  <div class="type-box">{{ item.type }}</div>
                  <div class="model-title">{{ item.group_name }}</div>
                </div>
              </div>
            </div>
            <div class="center-list">
              <div
                class="center-round-list"
                v-for="(item, index) in chartList.slice(1, 6)"
                :key="item"
              >
                <div class="center-item">
                  <img class="sit" src="../../assets/chart/sit.png" alt="" />
                  <img
                    class="in-arrow"
                    src="../../assets/chart/in-arrow.png"
                    alt=""
                  />
                  <img
                    class="out-arrow"
                    src="../../assets/chart/out-arrow.png"
                    alt=""
                    v-if="index != 4"
                  />
                  <img
                    class="stand"
                    src="../../assets/chart/stand.png"
                    alt=""
                  />
                </div>
              </div>
            </div>
            <div class="bottom-list">
              <div class="model-box" :class="{'chart-active': getStatus(chartList[0].group_name)}">
                <div
                  class="chart-bg"
                  v-if="getStatus(chartList[0].group_name)"
                ></div>
                <div class="type-box">{{ chartList[0].type }}</div>
                <div class="model-title">
                  {{ chartList[0].group_name.substr(0, 4) }}
                </div>
              </div>
            </div>
          </div>
        </div> -->
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import StepsView from '../StepsView.vue'
import ProcessView from '../ProcessView/index.vue'
import { ref, defineProps, defineExpose, defineEmits } from 'vue'
import { ArrowLeft, ArrowRight } from '@element-plus/icons-vue'
import { generateTitle } from '../../utils/i18n'
const getIcon = () => {
  return isCollapse.value ? ArrowLeft : ArrowRight
}
defineProps({
  reasoningList: {
    type: Array,
    default: []
  }
})
// const chartList = ref<Array<any>>([
//   {
//     type: 'G1',
//     group_name: 'Planner'
//   },
//   {
//     type: 'G2',
//     group_name: 'Retrieval'
//   },
//   {
//     type: 'G3',
//     group_name: 'Generate'
//   },
//   {
//     type: 'G4',
//     group_name: 'Converter'
//   },
//   {
//     type: 'G5',
//     group_name: 'Executor'
//   },
//   {
//     type: 'G6',
//     group_name: 'Optimize'
//   }
// ])
const emits = defineEmits(['changeStatusFun'])
// const getStatus = (val: any) => {
//   let arr: any = props.reasoningList
//   if (arr.length) {
//     let children: any = arr[arr.length - 1].children
//     if (!children.length) {
//       return false
//     }
//     let group_name = children[children.length - 1].group_name.toLowerCase()
//     return group_name === val.toLowerCase()
//   } else {
//     return false
//   }
// }
const changeStatusFun = () => {
  isCollapse.value = !isCollapse.value
  emits('changeStatusFun', isCollapse.value)
}
const isCollapse = ref<boolean>(true)
defineExpose({
  isCollapse
})
</script>

<style scoped>
@import './index.less';
</style>
