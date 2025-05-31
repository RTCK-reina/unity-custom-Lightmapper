using RTCK.NeuraBake.Runtime;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace RTCK.NeuraBake.Editor
{
    public class NeuraBakeWindow : EditorWindow
    {
        private NeuraBakeSettings settings;
        // RTCK.NeuraBakeWindows.cs 内
        // private string settingsFilePath = "Assets/RTCK_NeuraBake_Settings.json"; // 修正前
        private string settingsFilePath = "Assets/RTCK_NeuraBake/Settings/DefaultNeuraBakeSettings.json"; // 修正後 (例)
        private Vector2 scrollPosition;
        private bool isBaking = false;
        private CancellationTokenSource bakingCancellationSource;
        private float bakeProgress = 0f;
        private string bakeStatusMessage = "待機中";

        [MenuItem("RTCK/NeuraBake")]
        public static void ShowWindow()
        {
            NeuraBakeWindow window = GetWindow<NeuraBakeWindow>("NeuraBake");
            window.minSize = new Vector2(380, 580); // UI要素に合わせて調整
            window.Show();
        }

        private void OnEnable()
        {
            if (settings == null)
            {
                LoadSettings();
            }
        }

        private void OnDisable()
        {
            CancelBake();
        }

        private void OnGUI()
        {
            using (var scrollView = new EditorGUILayout.ScrollViewScope(scrollPosition))
            {
                scrollPosition = scrollView.scrollPosition;
                GUILayout.Label("RTCK NeuraBake 設定", EditorStyles.boldLabel);
                EditorGUILayout.Space();

                EditorGUI.BeginDisabledGroup(isBaking);
                DrawSettingsEditor();
                EditorGUI.EndDisabledGroup();

                EditorGUILayout.Space(20);
                DrawActionButtons();
                EditorGUILayout.Space(10);
                DrawStatusArea();
            }
        }

        private void DrawSettingsEditor()
        {
            if (settings == null)
            {
                EditorGUILayout.HelpBox("設定オブジェクトが初期化されていません。", MessageType.Error);
                return;
            }

            EditorGUI.BeginChangeCheck();

            // 全般設定
            GUILayout.Label("全般設定", EditorStyles.boldLabel);
            settings.resolution = EditorGUILayout.FloatField(new GUIContent("テクセル解像度", "ワールドユニットあたりのテクセル数。"), settings.resolution);
            settings.sampleCount = EditorGUILayout.IntSlider(new GUIContent("スーパーサンプリング", "1テクセルあたりのAAサンプル数。"), settings.sampleCount, 1, 64);

            EditorGUILayout.Space();
            // グローバルイルミネーション
            GUILayout.Label("グローバルイルミネーション", EditorStyles.boldLabel);
            settings.bounceCount = EditorGUILayout.IntSlider(new GUIContent("バウンス回数", "間接光の反射回数。"), settings.bounceCount, 0, 10);
            settings.skyIntensity = EditorGUILayout.Slider(new GUIContent("スカイライト強度", "環境光全体の明るさ係数。"), settings.skyIntensity, 0f, 5f);

            EditorGUILayout.Space();
            // アンビエントオクルージョン
            GUILayout.Label("アンビエントオクルージョン (AO)", EditorStyles.boldLabel);
            settings.useAmbientOcclusion = EditorGUILayout.Toggle(new GUIContent("AO 有効化", "オブジェクトの隙間や窪みに影を生成。"), settings.useAmbientOcclusion);
            if (settings.useAmbientOcclusion)
            {
                settings.aoSampleCount = EditorGUILayout.IntSlider(new GUIContent("AO サンプル数", "AOの品質。"), settings.aoSampleCount, 1, 128);
            }

            EditorGUILayout.Space();
            // シャドウ
            GUILayout.Label("シャドウ", EditorStyles.boldLabel);
            settings.shadowSamples = EditorGUILayout.IntSlider(new GUIContent("ソフトシャドウ サンプル", "1でハードシャドウ。値を上げるとソフトになるが重くなる。"), settings.shadowSamples, 1, 32);


            EditorGUILayout.Space();
            // 発光面
            GUILayout.Label("発光面 (Light Meshes)", EditorStyles.boldLabel);
            settings.emissiveBoost = EditorGUILayout.Slider(new GUIContent("発光ブースト", "NeuraBakeEmissiveSurfaceコンポーネントを持つオブジェクトの発光強度係数。"), settings.emissiveBoost, 0f, 10f);


            EditorGUILayout.Space();
            // ライトマップ出力
            GUILayout.Label("ライトマップ出力", EditorStyles.boldLabel);
            settings.directional = EditorGUILayout.Toggle(new GUIContent("指向性情報 (ベントノーマルY)", "ライトマップのAlphaチャンネルに簡易的な指向性情報（ベントノーマルY）を格納。"), settings.directional);
            settings.useDenoiser = EditorGUILayout.Toggle(new GUIContent("デノイザー有効化 (将来対応)", "ベイク結果のノイズを軽減。"), settings.useDenoiser);
            settings.atlasSize = EditorGUILayout.IntPopup(new GUIContent("テクスチャサイズ", "生成されるライトマップテクスチャの解像度。"), settings.atlasSize,
                new string[] { "256", "512", "1024", "2048", "4096", "8192" },
                new int[] { 256, 512, 1024, 2048, 4096, 8192 });

            if (EditorGUI.EndChangeCheck())
            {
                var (isValid, message) = settings.Validate();
                if (!isValid)
                {
                    EditorGUILayout.HelpBox(message, MessageType.Warning);
                }
            }
        }

        // DrawActionButtons, DrawStatusArea, SaveSettings, SaveSettingsAs, LoadSettings,
        // StartBakeAsync, CancelBake, SaveBakedLightmap は前回のコードから変更なしのため省略
        // (ただし、StartBakeAsync内のBakingCore呼び出しは新しいsettingsを渡すように)

        private void DrawActionButtons()
        {
            EditorGUILayout.LabelField("設定ファイル:", Path.GetFileName(settingsFilePath));
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("設定を保存")) SaveSettings();
                if (GUILayout.Button("別名で保存...")) SaveSettingsAs();
            }
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("設定を読み込み")) LoadSettings();
                if (GUILayout.Button("デフォルト設定にリセット"))
                {
                    if (EditorUtility.DisplayDialog("設定のリセット", "すべての設定を初期値に戻しますか？\n（現在の設定ファイルは上書きされません）", "はい", "いいえ"))
                    {
                        settings.Reset();
                        Repaint();
                    }
                }
            }

            EditorGUILayout.Space(20);

            GUI.backgroundColor = isBaking ? Color.red : Color.green;
            string bakeButtonText = isBaking ? "ベイク処理をキャンセル" : "ライトマップをベイク";
            if (GUILayout.Button(bakeButtonText, GUILayout.Height(40)))
            {
                if (isBaking)
                {
                    CancelBake();
                }
                else
                {
                    StartBakeAsync();
                }
            }
            GUI.backgroundColor = Color.white;
        }

        private void DrawStatusArea()
        {
            if (isBaking)
            {
                Rect r = EditorGUILayout.GetControlRect(false, EditorGUIUtility.singleLineHeight);
                EditorGUI.ProgressBar(r, bakeProgress, $"ベイク中: {bakeStatusMessage} ({Mathf.RoundToInt(bakeProgress * 100)}%)");
                EditorGUILayout.Space();
            }
            else
            {
                EditorGUILayout.HelpBox(bakeStatusMessage, MessageType.Info);
            }
        }

        private void SaveSettings()
        {
            if (settings == null) return;
            try
            {
                string json = JsonUtility.ToJson(settings, true);
                File.WriteAllText(settingsFilePath, json);
                Debug.Log($"NeuraBake: 設定を保存しました: {settingsFilePath}");
                AssetDatabase.Refresh();
            }
            catch (Exception e)
            {
                Debug.LogError($"NeuraBake: 設定の保存に失敗しました: {e.Message}");
            }
        }

        private void SaveSettingsAs()
        {
            if (settings == null) return;
            string directory = Path.GetDirectoryName(settingsFilePath);
            string fileName = Path.GetFileNameWithoutExtension(settingsFilePath);
            string path = EditorUtility.SaveFilePanel("設定を別名で保存", directory, fileName, "json");
            if (!string.IsNullOrEmpty(path))
            {
                settingsFilePath = path;
                SaveSettings();
            }
        }

        private void LoadSettings()
        {
            if (File.Exists(settingsFilePath))
            {
                try
                {
                    string json = File.ReadAllText(settingsFilePath);
                    settings = JsonUtility.FromJson<NeuraBakeSettings>(json);
                    if (settings == null)
                    {
                        Debug.LogWarning($"NeuraBake: 設定ファイル '{settingsFilePath}' の読み込みに失敗しました。新しい設定を作成します。");
                        settings = new NeuraBakeSettings();
                        settings.Reset();
                    }
                    else
                    {
                        Debug.Log($"NeuraBake: 設定を読み込みました: {settingsFilePath}");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"NeuraBake: 設定ファイルの読み込み中にエラーが発生しました: {e.Message}。新しい設定を作成します。");
                    settings = new NeuraBakeSettings();
                    settings.Reset();
                }
            }
            else
            {
                Debug.Log($"NeuraBake: 設定ファイルが見つかりません: {settingsFilePath}。新しい設定を作成します。");
                settings = new NeuraBakeSettings();
                settings.Reset();
            }
            Repaint();
        }

        private async void StartBakeAsync()
        {
            if (isBaking) return;
            if (settings == null)
            {
                EditorUtility.DisplayDialog("エラー", "設定がロードされていません。", "OK");
                return;
            }

            var (isValid, validationMessage) = settings.Validate();
            if (!isValid)
            {
                EditorUtility.DisplayDialog("設定エラー", $"設定値が不正です:\n{validationMessage}", "OK");
                return;
            }

            isBaking = true;
            bakeProgress = 0f;
            bakeStatusMessage = "初期化中...";
            bakingCancellationSource = new CancellationTokenSource();
            Repaint();

            try
            {
                BakingCore bakingCore = new BakingCore(settings);

                var progressReporter = new Progress<(float percentage, string message)>(update =>
                {
                    bakeProgress = update.percentage;
                    bakeStatusMessage = update.message;
                    Repaint();
                });

                bakeStatusMessage = "ライトマップ生成中...";
                Repaint();

                Texture2D lightmapTexture = await bakingCore.BakeLightmapAsync(bakingCancellationSource.Token, progressReporter);
                bakingCancellationSource.Token.ThrowIfCancellationRequested();

                if (lightmapTexture != null)
                {
                    bakeStatusMessage = "ライトマップ保存中...";
                    Repaint();
                    SaveBakedLightmap(lightmapTexture); // このメソッドは前回から変更なし
                    bakeStatusMessage = "ベイク処理完了";
                    DestroyImmediate(lightmapTexture);
                }
                else
                {
                    bakeStatusMessage = "ベイク処理は完了しましたが、テクスチャは生成されませんでした。";
                    Debug.LogWarning("NeuraBake: " + bakeStatusMessage);
                }
            }
            catch (OperationCanceledException)
            {
                bakeStatusMessage = "ベイク処理がキャンセルされました。";
                Debug.Log("NeuraBake: " + bakeStatusMessage);
            }
            catch (Exception e)
            {
                bakeStatusMessage = $"エラー: {e.Message}";
                Debug.LogError($"NeuraBake: ベイク処理中にエラーが発生しました: {e.Message}\n{e.StackTrace}");
            }
            finally
            {
                isBaking = false;
                bakeProgress = 0f;
                if (bakingCancellationSource != null)
                {
                    bakingCancellationSource.Dispose();
                    bakingCancellationSource = null;
                }
                Repaint();
            }
        }

        private void CancelBake()
        {
            if (isBaking && bakingCancellationSource != null && !bakingCancellationSource.IsCancellationRequested)
            {
                bakingCancellationSource.Cancel();
            }
        }

        private void SaveBakedLightmap(Texture2D texture)
        {
            if (texture == null) return;
            string sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            if (string.IsNullOrEmpty(sceneName)) sceneName = "UntitledScene";
            string directoryPath = $"Assets/NeuraBake_Lightmaps/{sceneName}";
            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }
            string fileName = $"{sceneName}_Lightmap_{System.DateTime.Now:yyyyMMdd_HHmmss}.exr";
            string outputPath = Path.Combine(directoryPath, fileName);

            try
            {
                byte[] bytes = texture.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
                File.WriteAllBytes(outputPath, bytes);
                AssetDatabase.Refresh();
                Debug.Log($"NeuraBake: ライトマップを保存しました: {outputPath}");
                TextureImporter importer = AssetImporter.GetAtPath(outputPath) as TextureImporter;
                if (importer != null)
                {
                    importer.textureType = TextureImporterType.Lightmap;
                    importer.sRGBTexture = false;
                    importer.mipmapEnabled = false;
                    importer.SaveAndReimport();
                    Debug.Log($"NeuraBake: {outputPath} のインポート設定を更新しました。");
                }
                Selection.activeObject = AssetDatabase.LoadAssetAtPath<Texture2D>(outputPath);
            }
            catch (Exception e)
            {
                Debug.LogError($"NeuraBake: ライトマップの保存に失敗しました ({outputPath}): {e.Message}");
            }
        }
    }
}