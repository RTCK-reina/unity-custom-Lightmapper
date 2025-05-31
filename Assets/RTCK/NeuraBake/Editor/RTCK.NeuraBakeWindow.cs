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
        // RTCK.NeuraBakeWindows.cs ��
        // private string settingsFilePath = "Assets/RTCK_NeuraBake_Settings.json"; // �C���O
        private string settingsFilePath = "Assets/RTCK_NeuraBake/Settings/DefaultNeuraBakeSettings.json"; // �C���� (��)
        private Vector2 scrollPosition;
        private bool isBaking = false;
        private CancellationTokenSource bakingCancellationSource;
        private float bakeProgress = 0f;
        private string bakeStatusMessage = "�ҋ@��";

        [MenuItem("RTCK/NeuraBake")]
        public static void ShowWindow()
        {
            NeuraBakeWindow window = GetWindow<NeuraBakeWindow>("NeuraBake");
            window.minSize = new Vector2(380, 580); // UI�v�f�ɍ��킹�Ē���
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
                GUILayout.Label("RTCK NeuraBake �ݒ�", EditorStyles.boldLabel);
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
                EditorGUILayout.HelpBox("�ݒ�I�u�W�F�N�g������������Ă��܂���B", MessageType.Error);
                return;
            }

            EditorGUI.BeginChangeCheck();

            // �S�ʐݒ�
            GUILayout.Label("�S�ʐݒ�", EditorStyles.boldLabel);
            settings.resolution = EditorGUILayout.FloatField(new GUIContent("�e�N�Z���𑜓x", "���[���h���j�b�g������̃e�N�Z�����B"), settings.resolution);
            settings.sampleCount = EditorGUILayout.IntSlider(new GUIContent("�X�[�p�[�T���v�����O", "1�e�N�Z���������AA�T���v�����B"), settings.sampleCount, 1, 64);

            EditorGUILayout.Space();
            // �O���[�o���C���~�l�[�V����
            GUILayout.Label("�O���[�o���C���~�l�[�V����", EditorStyles.boldLabel);
            settings.bounceCount = EditorGUILayout.IntSlider(new GUIContent("�o�E���X��", "�Ԑڌ��̔��ˉ񐔁B"), settings.bounceCount, 0, 10);
            settings.skyIntensity = EditorGUILayout.Slider(new GUIContent("�X�J�C���C�g���x", "�����S�̖̂��邳�W���B"), settings.skyIntensity, 0f, 5f);

            EditorGUILayout.Space();
            // �A���r�G���g�I�N���[�W����
            GUILayout.Label("�A���r�G���g�I�N���[�W���� (AO)", EditorStyles.boldLabel);
            settings.useAmbientOcclusion = EditorGUILayout.Toggle(new GUIContent("AO �L����", "�I�u�W�F�N�g�̌��Ԃ�E�݂ɉe�𐶐��B"), settings.useAmbientOcclusion);
            if (settings.useAmbientOcclusion)
            {
                settings.aoSampleCount = EditorGUILayout.IntSlider(new GUIContent("AO �T���v����", "AO�̕i���B"), settings.aoSampleCount, 1, 128);
            }

            EditorGUILayout.Space();
            // �V���h�E
            GUILayout.Label("�V���h�E", EditorStyles.boldLabel);
            settings.shadowSamples = EditorGUILayout.IntSlider(new GUIContent("�\�t�g�V���h�E �T���v��", "1�Ńn�[�h�V���h�E�B�l���グ��ƃ\�t�g�ɂȂ邪�d���Ȃ�B"), settings.shadowSamples, 1, 32);


            EditorGUILayout.Space();
            // ������
            GUILayout.Label("������ (Light Meshes)", EditorStyles.boldLabel);
            settings.emissiveBoost = EditorGUILayout.Slider(new GUIContent("�����u�[�X�g", "NeuraBakeEmissiveSurface�R���|�[�l���g�����I�u�W�F�N�g�̔������x�W���B"), settings.emissiveBoost, 0f, 10f);


            EditorGUILayout.Space();
            // ���C�g�}�b�v�o��
            GUILayout.Label("���C�g�}�b�v�o��", EditorStyles.boldLabel);
            settings.directional = EditorGUILayout.Toggle(new GUIContent("�w������� (�x���g�m�[�}��Y)", "���C�g�}�b�v��Alpha�`�����l���ɊȈՓI�Ȏw�������i�x���g�m�[�}��Y�j���i�[�B"), settings.directional);
            settings.useDenoiser = EditorGUILayout.Toggle(new GUIContent("�f�m�C�U�[�L���� (�����Ή�)", "�x�C�N���ʂ̃m�C�Y���y���B"), settings.useDenoiser);
            settings.atlasSize = EditorGUILayout.IntPopup(new GUIContent("�e�N�X�`���T�C�Y", "��������郉�C�g�}�b�v�e�N�X�`���̉𑜓x�B"), settings.atlasSize,
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
        // StartBakeAsync, CancelBake, SaveBakedLightmap �͑O��̃R�[�h����ύX�Ȃ��̂��ߏȗ�
        // (�������AStartBakeAsync����BakingCore�Ăяo���͐V����settings��n���悤��)

        private void DrawActionButtons()
        {
            EditorGUILayout.LabelField("�ݒ�t�@�C��:", Path.GetFileName(settingsFilePath));
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("�ݒ��ۑ�")) SaveSettings();
                if (GUILayout.Button("�ʖ��ŕۑ�...")) SaveSettingsAs();
            }
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("�ݒ��ǂݍ���")) LoadSettings();
                if (GUILayout.Button("�f�t�H���g�ݒ�Ƀ��Z�b�g"))
                {
                    if (EditorUtility.DisplayDialog("�ݒ�̃��Z�b�g", "���ׂĂ̐ݒ�������l�ɖ߂��܂����H\n�i���݂̐ݒ�t�@�C���͏㏑������܂���j", "�͂�", "������"))
                    {
                        settings.Reset();
                        Repaint();
                    }
                }
            }

            EditorGUILayout.Space(20);

            GUI.backgroundColor = isBaking ? Color.red : Color.green;
            string bakeButtonText = isBaking ? "�x�C�N�������L�����Z��" : "���C�g�}�b�v���x�C�N";
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
                EditorGUI.ProgressBar(r, bakeProgress, $"�x�C�N��: {bakeStatusMessage} ({Mathf.RoundToInt(bakeProgress * 100)}%)");
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
                Debug.Log($"NeuraBake: �ݒ��ۑ����܂���: {settingsFilePath}");
                AssetDatabase.Refresh();
            }
            catch (Exception e)
            {
                Debug.LogError($"NeuraBake: �ݒ�̕ۑ��Ɏ��s���܂���: {e.Message}");
            }
        }

        private void SaveSettingsAs()
        {
            if (settings == null) return;
            string directory = Path.GetDirectoryName(settingsFilePath);
            string fileName = Path.GetFileNameWithoutExtension(settingsFilePath);
            string path = EditorUtility.SaveFilePanel("�ݒ��ʖ��ŕۑ�", directory, fileName, "json");
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
                        Debug.LogWarning($"NeuraBake: �ݒ�t�@�C�� '{settingsFilePath}' �̓ǂݍ��݂Ɏ��s���܂����B�V�����ݒ���쐬���܂��B");
                        settings = new NeuraBakeSettings();
                        settings.Reset();
                    }
                    else
                    {
                        Debug.Log($"NeuraBake: �ݒ��ǂݍ��݂܂���: {settingsFilePath}");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"NeuraBake: �ݒ�t�@�C���̓ǂݍ��ݒ��ɃG���[���������܂���: {e.Message}�B�V�����ݒ���쐬���܂��B");
                    settings = new NeuraBakeSettings();
                    settings.Reset();
                }
            }
            else
            {
                Debug.Log($"NeuraBake: �ݒ�t�@�C����������܂���: {settingsFilePath}�B�V�����ݒ���쐬���܂��B");
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
                EditorUtility.DisplayDialog("�G���[", "�ݒ肪���[�h����Ă��܂���B", "OK");
                return;
            }

            var (isValid, validationMessage) = settings.Validate();
            if (!isValid)
            {
                EditorUtility.DisplayDialog("�ݒ�G���[", $"�ݒ�l���s���ł�:\n{validationMessage}", "OK");
                return;
            }

            isBaking = true;
            bakeProgress = 0f;
            bakeStatusMessage = "��������...";
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

                bakeStatusMessage = "���C�g�}�b�v������...";
                Repaint();

                Texture2D lightmapTexture = await bakingCore.BakeLightmapAsync(bakingCancellationSource.Token, progressReporter);
                bakingCancellationSource.Token.ThrowIfCancellationRequested();

                if (lightmapTexture != null)
                {
                    bakeStatusMessage = "���C�g�}�b�v�ۑ���...";
                    Repaint();
                    SaveBakedLightmap(lightmapTexture); // ���̃��\�b�h�͑O�񂩂�ύX�Ȃ�
                    bakeStatusMessage = "�x�C�N��������";
                    DestroyImmediate(lightmapTexture);
                }
                else
                {
                    bakeStatusMessage = "�x�C�N�����͊������܂������A�e�N�X�`���͐�������܂���ł����B";
                    Debug.LogWarning("NeuraBake: " + bakeStatusMessage);
                }
            }
            catch (OperationCanceledException)
            {
                bakeStatusMessage = "�x�C�N�������L�����Z������܂����B";
                Debug.Log("NeuraBake: " + bakeStatusMessage);
            }
            catch (Exception e)
            {
                bakeStatusMessage = $"�G���[: {e.Message}";
                Debug.LogError($"NeuraBake: �x�C�N�������ɃG���[���������܂���: {e.Message}\n{e.StackTrace}");
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
                Debug.Log($"NeuraBake: ���C�g�}�b�v��ۑ����܂���: {outputPath}");
                TextureImporter importer = AssetImporter.GetAtPath(outputPath) as TextureImporter;
                if (importer != null)
                {
                    importer.textureType = TextureImporterType.Lightmap;
                    importer.sRGBTexture = false;
                    importer.mipmapEnabled = false;
                    importer.SaveAndReimport();
                    Debug.Log($"NeuraBake: {outputPath} �̃C���|�[�g�ݒ���X�V���܂����B");
                }
                Selection.activeObject = AssetDatabase.LoadAssetAtPath<Texture2D>(outputPath);
            }
            catch (Exception e)
            {
                Debug.LogError($"NeuraBake: ���C�g�}�b�v�̕ۑ��Ɏ��s���܂��� ({outputPath}): {e.Message}");
            }
        }
    }
}