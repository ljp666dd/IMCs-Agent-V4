<template>
  <div class="reasoning-box">
    <div
      class="problem-item"
      v-for="(key, num) in props.reasoningList"
      :key="num"
      :id="`${num}title`"
    >
      <!-- <div class="title-box">
        <p>{{ generateTitle("问题设定") }}:</p>
        <span>{{ key.title }}</span>
      </div> -->
      <div
        class="reasoning-item"
        v-for="(item, index) in key.children"
        :key="index"
        :id="`${num}${index}content`"
      >
        <div class="title">
          <img src="../../assets/logo.png" alt="" />
          <!-- <span v-if="item.source === 'Planner'"
            >{{ generateTitle('角色') }}（{{ item.source }}）</span
          >
          <span v-else
            >{{ generateTitle('模型名称') }}（{{ `${item.source}` }}）</span
          > -->
          <span> {{ `${item.source}` }} </span>
          <!-- <span v-else>{{generateTitle('模型名称')}}（{{ `${item.source}: ${item.agent_name}` }}）</span> -->
          <p
            class="active_item"
            v-if="key.children.length - 1 === index && reasonStatus.show"
          >
            <img src="../../assets/layout/vector.png" alt="" />
            {{ generateTitle("推理中") }}...
          </p>
        </div>
        <div class="reasoning-content">
          <div
            class="reasoning-text"
            :class="{
              active: key.children.length - 1 === index && reasonStatus.show,
            }"
          >
            <div
              style="font-size: 14px"
              :class="{ 'show-box': current[index] }"
            >
              <MdPreview
                style="
                  background: linear-gradient(
                    to bottom,
                    #d5e6f4 0%,
                    #afceea 50%,
                    #c2d9ef 100%
                  );
                "
                :editorId="id"
                :modelValue="item.content"
              />
            </div>
          </div>
          <div class="icon-box">
            <el-icon color="#fff" @click="changeShow(index)"
              ><component :is="getIcon(index)"></component
            ></el-icon>
          </div>
        </div>
      </div>
      <div
        class="complete-box"
        v-if="completeList[num].show && num === completeList[num].index"
      >
        {{
          generateTitle(
            "推理完成，如果希望了解更多可继续在下面框中提问或新建对话。"
          )
        }}
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, defineExpose, defineEmits, defineProps } from "vue";
import { Plus, Minus } from "@element-plus/icons-vue";
import { MdPreview } from "md-editor-v3";
import { generateTitle } from "../../utils/i18n";
import "md-editor-v3/lib/style.css";
const props: any = defineProps({
  reasoningList: {
    type: Array,
    default: [],
  },
});
const id = "preview-only";
const emits = defineEmits(["completeFun"]);
const current = ref<Array<any>>([]);
const reasonStatus = ref({
  index: 0,
  show: false,
});
const completeList = ref<Array<any>>([]);
const getIcon = (index: number) => {
  return current.value[index] ? Plus : Minus;
};
const changeShow = (index: number) => {
  current.value[index] = !current.value[index];
};
// const reasoningList = ref<Array<any>>([])
const getAnswer = async () => {
  emits("completeFun");
};
// function extractSynthesisProcess() {
//   talkList.value
//     .split(">>>>>>>> USING AUTO REPLY...")
//     .forEach((item: any, index: number) => {
//       if (index) {
//         let data = item.split(
//           "\nsynthesis_scientist (to Generate_Group_Admin):\n"
//         )[1];
//         if (data) {
//           reasoningList.value[reasoningList.value.length - 1].content = data;
//         }
//       } else {
//         let data = item.split("Next speaker: ");
//         reasoningList.value[
//           reasoningList.value.length - 1
//         ].title = `模型名称（${data[data.length - 1].replace(/\n/g, "")}）`;
//       }
//     });
// }

// const getData = (e: any) => {
//   let regex = /Next speaker: (.*)/;
//   let regexnew = /\(to (.*)\)/;
//   reasonStatus.value.show = true;
//   if (e.includes(">>>>>>>> USING AUTO REPLY...")) {
//     reasoningList.value[reasoningList.value.length - 1].title = oldLine.value[oldLine.value.length - 1].match(regex)[1];
//     startStatus.value = true;
//   } else if (startStatus.value) {
//     if (
//       e.includes(
//         `${reasoningList.value[reasoningList.value.length - 1].title} (to`
//       )
//     ) {
//       reasoningList.value[reasoningList.value.length - 1].title = e.match(regexnew)[1];
//     } else {
//       console.log(e,e.split("**TERMINATE**"),'111');

//       if (e.split("**TERMINATE**")[1]) {
//         reasoningList.value[reasoningList.value.length - 1].content +=
//           e.split("**TERMINATE**")[0];
//         reasonStatus.value.show = false;
//         endShow.value = true;
//       } else if (e.split("`TERMINATE`")[1]) {
//         reasoningList.value[reasoningList.value.length - 1].content +=
//           e.split("`TERMINATE`")[0];
//         reasonStatus.value.show = false;
//         endShow.value = true;
//       } else {
//         reasoningList.value[reasoningList.value.length - 1].content += e;
//       }
//     }
//   }
//   oldLine.value.push(e);

// }
defineExpose({
  getAnswer,
  reasonStatus,
  completeList,
});
</script>
<style scoped>
@import "./index.less";
</style>
