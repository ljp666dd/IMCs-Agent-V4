<template>
  <div>
    <el-dropdown class="language-box" placement="bottom-start">
      <el-button>{{
        state.curLanguage === "en" ? state.curLanguage : "中"
      }}</el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item @click="zh('zh-cn')">中文</el-dropdown-item>
          <el-dropdown-item @click="en('en')">English</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>
<script setup lang='ts'>
import { reactive } from "vue";
import { useI18n } from "vue-i18n";
const { locale } = useI18n();

const state = reactive({
  curLanguage: "zh-cn",
});
const zh = (e: any) => {
  state.curLanguage = e;
  locale.value = state.curLanguage;
};
const en = (e: any) => {
  state.curLanguage = e;
  locale.value = state.curLanguage;
};
</script>
<style scoped lang='less'>
.language-box {
  position: fixed;
  top: 6px;
  left: 32px;
  .el-button {
    background: none;
    color: #fff;
    font-size: 14px;
    width: 24px;
    padding: 0;
    height: 24px;
    border-radius: 50%;
    &:focus {
      outline: none;
    }
    &:hover {
      border: 1px solid #3ba4ff;
      color: #3ba4ff;
    }
  }
}
</style>
