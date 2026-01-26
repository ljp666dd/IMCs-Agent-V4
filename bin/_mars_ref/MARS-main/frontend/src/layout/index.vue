<template>
  <div class="common-layout">
    <el-container>
      <el-main style="padding: 40px 32px 32px">
        <router-view></router-view>
      </el-main>
      <el-footer style="height: 40px">
        <div class="feeter-box">{{ generateTitle('智能协作无界，材料创新无限') }}</div>
      </el-footer>
    </el-container>
  </div>
</template>
<script setup lang="ts">
import {generateTitle} from '../utils/i18n'
</script>
<style scoped>
@import "./index.less";
</style>
